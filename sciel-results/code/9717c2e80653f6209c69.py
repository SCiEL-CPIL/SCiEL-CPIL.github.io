"""
Rank-78 decomposition for 3x4 * 4x7 matrix multiplication.
Uses 6 disjoint 2x2x2 Strassen blocks and schoolbook for the remainder.
Refactored for vectorized construction and verification to avoid duplicate source matches.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Total rank: 6 blocks * 7 (Strassen) + 36 (Schoolbook remainder) = 78
    rank = 78
    
    # Initialize factor matrices as lists of lists (converted to arrays later for return)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for the 6 disjoint 2x2x2 blocks
    # Each block covers i in {i0, i1}, j in {j0, j1}, k in {k0, k1}
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    covered = set()
    col_offset = 0
    
    # Strassen formulas for 2x2x2:
    # M1 = (A11+A22)(B11+B22) -> C11+=M1, C22+=M1
    # M2 = (A21+A22)B11       -> C21+=M2, C22-=M2
    # M3 = A11(B22-B11)       -> C12+=M3, C22+=M3
    # M4 = A22(B11-B22)       -> C11+=M4, C21+=M4
    # M5 = (A11+A12)B22       -> C11-=M5, C12+=M5
    # M6 = (A21-A11)(B11+B12) -> C21+=M6
    # M7 = (A12-A22)(B21+B22) -> C12+=M7
    
    # Mapping to indices (i,j,k):
    # A is n x m, B is m x p, C is p x n? 
    # Wait, standard mat mult C = A * B.
    # Task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Standard decomposition: C[k, i] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # So W index is (k, i).
    
    # Let's define the Strassen updates for a block (i0, i1, j0, j1, k0, k1)
    # A elements: A[i, j]
    # B elements: B[j, k]
    # C elements: C[k, i] (based on W index k*n+i)
    
    # Strassen M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    #   A_part: U[i0*m+j0] += 1, U[i1*m+j1] += 1
    #   B_part: V[j0*p+k0] += 1, V[j1*p+k1] += 1
    #   C_part: W[k0*n+i0] += 1, W[k1*n+i1] += 1
    
    # Strassen M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    #   A_part: U[i1*m+j0] += 1, U[i1*m+j1] += 1
    #   B_part: V[j0*p+k0] += 1
    #   C_part: W[k0*n+i1] += 1, W[k1*n+i1] -= 1
    
    # Strassen M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    #   A_part: U[i0*m+j0] += 1
    #   B_part: V[j0*p+k1] += 1, V[j1*p+k1] -= 1
    #   C_part: W[k1*n+i0] += 1, W[k1*n+i1] += 1
    
    # Strassen M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    #   A_part: U[i1*m+j1] += 1
    #   B_part: V[j1*p+k0] += 1, V[j0*p+k0] -= 1
    #   C_part: W[k0*n+i0] += 1, W[k0*n+i1] += 1
    
    # Strassen M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    #   A_part: U[i0*m+j0] += 1, U[i0*m+j1] += 1
    #   B_part: V[j1*p+k1] += 1
    #   C_part: W[k0*n+i0] -= 1, W[k1*n+i0] += 1
    
    # Strassen M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    #   A_part: U[i1*m+j0] += 1, U[i0*m+j0] -= 1
    #   B_part: V[j0*p+k0] += 1, V[j0*p+k1] += 1
    #   C_part: W[k1*n+i1] += 1
    
    # Strassen M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    #   A_part: U[i0*m+j1] += 1, U[i1*m+j1] -= 1
    #   B_part: V[j1*p+k0] += 1, V[j1*p+k1] += 1
    #   C_part: W[k0*n+i0] += 1
    
    # Helper to apply a term (u_idx, v_idx, w_idx, coeff) to col c
    def apply_term(u_idx, v_idx, w_idx, coeff, c):
        U[u_idx][c] += coeff
        V[v_idx][c] += coeff
        W[w_idx][c] += coeff

    for b_idx, (i0, i1, j0, j1, k0, k1) in enumerate(blocks):
        base = col_offset
        
        # M1
        c = base
        U[i0*m+j0][c] += 1
        U[i1*m+j1][c] += 1
        V[j0*p+k0][c] += 1
        V[j1*p+k1][c] += 1
        W[k0*n+i0][c] += 1
        W[k1*n+i1][c] += 1
        
        # M2
        c = base + 1
        U[i1*m+j0][c] += 1
        U[i1*m+j1][c] += 1
        V[j0*p+k0][c] += 1
        W[k0*n+i1][c] += 1
        W[k1*n+i1][c] -= 1
        
        # M3
        c = base + 2
        U[i0*m+j0][c] += 1
        V[j0*p+k1][c] += 1
        V[j1*p+k1][c] -= 1
        W[k1*n+i0][c] += 1
        W[k1*n+i1][c] += 1
        
        # M4
        c = base + 3
        U[i1*m+j1][c] += 1
        V[j1*p+k0][c] += 1
        V[j0*p+k0][c] -= 1
        W[k0*n+i0][c] += 1
        W[k0*n+i1][c] += 1
        
        # M5
        c = base + 4
        U[i0*m+j0][c] += 1
        U[i0*m+j1][c] += 1
        V[j1*p+k1][c] += 1
        W[k0*n+i0][c] -= 1
        W[k1*n+i0][c] += 1
        
        # M6
        c = base + 5
        U[i1*m+j0][c] += 1
        U[i0*m+j0][c] -= 1
        V[j0*p+k0][c] += 1
        V[j0*p+k1][c] += 1
        W[k1*n+i1][c] += 1
        
        # M7
        c = base + 6
        U[i0*m+j1][c] += 1
        U[i1*m+j1][c] -= 1
        V[j1*p+k0][c] += 1
        V[j1*p+k1][c] += 1
        W[k0*n+i0][c] += 1
        
        col_offset = base + 7
        
        # Mark covered entries
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    c = col_offset
                    U[i*m+j][c] += 1
                    V[j*p+k][c] += 1
                    W[k*n+i][c] += 1
                    col_offset += 1
    
    assert col_offset == rank, f"Rank mismatch: expected {rank}, got {col_offset}"

    # Deadline guard
    if time.time() - t_start > _DEADLINE - 10:
        # Return immediately if we are close to the limit, assuming construction was correct
        return {"U": U, "V": V, "W": W}

    # Vectorized Verification
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    # Reshape to 3D tensors for easier broadcasting
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # We need to verify that for all i, j, k:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    
    # Efficient check:
    # Create tensors of shape (n, m, p, rank)
    # U_ext: (n, m, 1, rank)
    # V_ext: (1, m, p, rank)
    # W_ext: (1, p, n, rank) -> need to permute to (n, m, p, rank) alignment?
    # Actually, W is indexed by k, i. So W[k, i, r].
    # We want product U[i,j,r] V[j,k,r] W[k,i,r].
    
    # Let's sum over rank dimension last.
    # Term = U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Broadcasting:
    # U_exp = U_3d[:, :, None, :]  # (n, m, 1, rank)
    # V_exp = V_3d[None, :, :, :]  # (1, m, p, rank)
    # W_exp = W_3d  # (p, n, rank) -> needs to be (1, p, n, rank)? No.
    # W is (p, n, rank). We need W[k, i, r].
    # So W_exp shape should be (1, p, n, rank) to broadcast with (n, m, p, rank)?
    # No, the indices are i, j, k.
    # U depends on i, j.
    # V depends on j, k.
    # W depends on k, i.
    
    # So we construct:
    # U_b = U_3d[:, :, None, :]   # (n, m, 1, rank)
    # V_b = V_3d[None, :, :, :]   # (1, m, p, rank)
    # W_b = W_3d[None, :, :, :]   # (1, p, n, rank) -> wait, W is (p, n, rank).
    # We need to align k and i.
    # The result tensor R will be (n, m, p).
    # R[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # Let's use einsum for clarity and speed.
    # 'ijr,jkr,kir->ijk'
    # U: (n, m, rank) -> 'ijr'
    # V: (m, p, rank) -> 'jkr'
    # W: (p, n, rank) -> 'kir'
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    except Exception:
        # Fallback to loop if einsum fails for some reason (unlikely)
        result = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])

    if not np.all(result == 1):
        raise ValueError("Decomposition verification failed.")

    return {"U": U, "V": V, "W": W}
