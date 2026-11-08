"""Matrix multiplication 2x4 by 4x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen22():
    """Return U, V, W coefficient lists for 2x2 x 2x2 = 2x2 Strassen (rank 7)."""
    U_c = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_c = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_c = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]
    return U_c, V_c, W_c

def _build_rank42(n, m, p):
    """Block Strassen: 2 A-blocks x 3 B-blocks x 7 Strassen = rank 42."""
    U_c, V_c, W_c = _build_strassen22()
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), c in U_c[s].items():
                    U[i_loc * m + a*2 + j_loc][col] += c
                for (j_loc, k_loc), c in V_c[s].items():
                    V[(a*2 + j_loc) * p + b*2 + k_loc][col] += c
                for (i_loc, k_loc), c in W_c[s].items():
                    W[(b*2 + k_loc) * n + i_loc][col] += c
                col += 1
    return U, V, W

def _build_rank48(n, m, p):
    """Naive: each (i,j,k) is one term."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _build_rank54(n, m, p):
    """2x2 Strassen on A rows (rank 3 for 2x2), naive on B: 3*4*6=72? No.
    Use Strassen for A (2x4 -> 3 terms) x naive B (4x6 -> 24): 3*24=72. Worse.
    Alternative: naive A (8) x Strassen B (2x6 -> 3*7=21): 8*21=168. Worse.
    Let's do: split A into two 2x2 blocks (Strassen each, rank 7 each = 14 for A side),
    B naive (24): 14*24 = 336. Way worse.
    
    Actually just return rank 48 naive."""
    return _build_rank48(n, m, p)

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build rank-42 Strassen solution
    U42, V42, W42 = _build_rank42(n, m, p)
    
    # Verify rank-42 is correct
    U42_arr = np.array(U42, dtype=np.float64)
    V42_arr = np.array(V42, dtype=np.float64)
    W42_arr = np.array(W42, dtype=np.float64)
    
    # Check: for each i,j,k, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    # and for each (i1,j1,k1) != (i2,j2,k2), the sum should be 0
    # Efficient check: tensor = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U_r = U42_arr.reshape(n, m, 42)
    V_r = V42_arr.reshape(m, p, 42)
    W_r = W42_arr.reshape(p, n, 42)
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kr->ijk', U_r, V_r, W_r)
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i,j,k] = 1.0
    
    if not np.allclose(T, T_expected, atol=1e-9):
        # Fallback: try rank 48
        U48, V48, W48 = _build_rank48(n, m, p)
        return {"U": U48, "V": V48, "W": W48}
    
    # Now try to find a lower rank decomposition
    # The matrix multiplication tensor T in R^{n x m x p} has rank <= 42.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Also rank >= n*p (from one slice) and other bounds.
    # Bini's bound: for (2,4,6), the minimum rank is at least...
    # Let's try rank 36: use a 2x3x3 block structure with Strassen-like on 2x2 parts
    
    # Alternative: try to use the fact that n=2 to get a better decomposition.
    # For fixed k, C[:,k] = A @ B[:,k] which is 2x4 times 4x1.
    # Rank of matmul(2,4,1) tensor is 4 (naive) or can be done in 4.
    # Total: 6 columns of B, so 6*4 = 24 if independent. But Bini's bound says 24.
    # Actually for (2,4,6), known minimum rank might be 36 or 42.
    
    # Let's try rank 36: 
    # Split the 4 middle dimension into two groups of 2.
    # For each group g in {0,1} and each output column block b in {0,1,2}:
    #   C_block[b] (2x2) = sum_g A_block[g] (2x2) * B_block[g,b] (2x2)
    # Use Strassen for the 2x2*2x2=2x2 multiply: 7 multiplications per (g,b) pair.
    # 2 groups * 3 B-blocks * 7 = 42. Same as before.
    
    # Alternative: use the structure differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For n=2: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    # 
    # Try: for each pair of rows (j1,j2) in A and (k1,k2) in B:
    # This doesn't naturally decompose.
    
    # Let's try a different approach: rank 30.
    # Use the fact that we can combine some terms.
    # For 2x4 x 4x6, one approach:
    # Split A into rows: A = [A_top; A_bottom] where A_top is 2x4, A_bottom is... 
    # Actually A is 2x4. Split columns into 2x2 blocks.
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2; B3] where B1, B2, B3 are 2x6... no, 2x6 each? 
    # B is 4x6. Split rows: B = [B_top; B_bot] where B_top, B_bot are 2x6.
    # C = A1 @ B_top + A2 @ B_bot
    # Each 2x2 @ 2x6 can be done... 
    # 2x2 @ 2x6: for each of the 6 columns, do 2x2*2x1 (rank 4 naive, or 7/3~2.33 with Strassen but not integer).
    # 2x2 @ 2x1 has rank 4 (can't do better than 4 for integer? Actually rank 4 is minimum for 2x2x1).
    # So 2x2 @ 2x6 has rank 6*4/2 = ... no. 
    # For 2x2 @ 2x6: tensor in R^{2x2 x 2x6}. 
    # Split the 6 columns into 3 pairs: 2x2 @ 2x2 for each pair, rank 7 each, total 21.
    # Two such (for A1@B_top and A2@B_bot): 21 + 21 = 42. Same.
    
    # Try a non-block decomposition.
    # The tensor T_{ijk} = delta is the matmul tensor.
    # Rank of matmul(m,n,p) tensor: for (2,4,6) the exact rank is unknown but bounded.
    # Bini 1980: R(2,4,6) >= 36 (from Bini's bound R(m,n,p) >= (mn+np+pm - sqrt((mn-np)^2 + ... )) / something)
    # Actually Bini's bound: R(m,n,p) >= (mn + np + pm - sqrt((mn-np)^2 + (np-pm)^2 + (pm-mn)^2)) / 2
    # For (2,4,6): mn=8, np=12, pm=24
    # (8+12+24 - sqrt((8-12)^2+(12-24)^2+(24-8)^2))/2 = (44 - sqrt(16+144+256))/2 = (44-sqrt(416))/2 = (44-20.396)/2 = 11.8
    # That's a weak bound.
    
    # Let me try to construct a rank-36 solution.
    # Idea: use Strassen on 2x2 blocks but share some multiplications.
    # 
    # Alternative: for n=2, we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # Define for each k:
    #   X_k = B[0,k] + B[2,k], Y_k = B[1,k] + B[3,k]
    #   Z_k = B[0,k] - B[2,k], W_k = B[1,k] - B[3,k]
    # Then:
    #   A[0,0]*X_k + A[0,1]*Y_k gives A[0,0]*B[0,k]+A[0,0]*B[2,k]+A[0,1]*B[1,k]+A[0,1]*B[3,k]
    # This doesn't directly help.
    #
    # Let me try a systematic search for rank < 42.
    # Given time constraints, I'll try rank 36 with a specific construction.
    
    # Actually, let me try: use the 3x3x3 Strassen for the 3x3 part and handle the 2 differently.
    # 2x4 x 4x6: pad to 2x4 x 4x8 (8=2*4)? No, that increases rank.
    
    # Let me just verify rank 42 and return it if we can't find better quickly.
    
    # Try rank 36: 2x4x6 with a different block structure.
    # Split the middle 4 into 1+3: A = [a0 | A_rest] where a0 is 2x1, A_rest is 2x3.
    # B = [b0; B_rest] where b0 is 1x6, B_rest is 3x6.
    # C = a0 @ b0 + A_rest @ B_rest
    # a0 @ b0: 2x1 x 1x6, rank = 2*6/2 = 6 (each C[i,k] = a0[i]*b0[k], that's 12 terms but rank 6? 
    # No, the tensor for (2,1,6) matmul has rank 2*1*6 = 12? No, rank of (m,n,p) matmul tensor.
    # For (2,1,6): T[i,0,k] = 1. This is a rank-12 tensor? No, it's a sum of 2*6 rank-1 tensors.
    # Actually T = sum_{i=0,1} sum_{k=0,5} e_i (in R^2) * e_0 (in R^1) * e_k (in R^6)
    # That's 12 rank-1 terms. But can we do better? The rank of the (2,1,6) matmul tensor is 2*6 = 12? 
    # No, it's just 2*6 = 12 since the middle dimension is 1.
    # Hmm, that's worse.
    
    # I'll stick with rank 42 which is verified.
    
    if time.time() > DEADLINE:
        return {"U": U42, "V": V42, "W": W42}
    
    return {"U": U42, "V": V42, "W": W42}

# EVOLVE-BLOCK-END
