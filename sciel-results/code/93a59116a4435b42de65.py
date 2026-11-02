"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Uses a block-Strassen approach: split the 4x4 middle dimension into four 2x2 blocks,
apply Strassen (rank 7) to the 2x2x2 sub-problems, and schoolbook for the remaining
2x2x1 and 2x1x2 cross terms. This yields rank 36.

Then attempts to reduce the rank further by checking if any columns are linearly
redundant (i.e., can be expressed as a linear combination of other columns across
all three factor matrices simultaneously).
"""
import numpy as np
import time

def _build_block_strassen(n, m, p):
    """
    Build a block-Strassen decomposition for n x m times m x p.
    
    For n=2, m=4, p=5:
    - Split m=4 into two blocks of 2: m1=2, m2=2
    - A is 2x4, split into A1 (2x2) and A2 (2x2)
    - B is 4x5, split into B1 (2x5) and B2 (2x5)
    - C = A1*B1 + A2*B2 (both 2x5)
    
    For each 2x2 * 2x5 product:
    - Split the 5 columns of B into: 2+2+1
    - The 2x2 * 2x2 part uses Strassen (rank 7)
    - The 2x2 * 2x1 part uses schoolbook (rank 4)
    - The 2x1 * 2x1 part... wait, that's not right.
    
    Let me reconsider. We have A (2x4) * B (4x5).
    
    Split B's rows into B_top (2x5) and B_bot (2x5).
    Split A's columns into A_left (2x2) and A_right (2x2).
    
    C = A_left * B_top + A_right * B_bot
    
    For A_left * B_top: (2x2) * (2x5)
    Split B_top's columns into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x1)
    A_left * B_top_left: 2x2 * 2x2 -> Strassen rank 7
    A_left * B_top_mid: 2x2 * 2x2 -> Strassen rank 7  
    A_left * B_top_right: 2x2 * 2x1 -> schoolbook rank 4
    
    Similarly for A_right * B_bot:
    A_right * B_bot_left: 2x2 * 2x2 -> Strassen rank 7
    A_right * B_bot_mid: 2x2 * 2x2 -> Strassen rank 7
    A_right * B_bot_right: 2x2 * 2x1 -> schoolbook rank 4
    
    Total: 7+7+4 + 7+7+4 = 36
    
    But wait, we need to be careful about how the tensor indices work.
    
    The tensor T[a, b, c] where:
    - a = i*m + j (i in 0..n-1, j in 0..m-1)
    - b = j'*p + k (j' in 0..m-1, k in 0..p-1)
    - c = k'*n + i' (k' in 0..p-1, i' in 0..n-1)
    
    T[a, b, c] = 1 iff i=i', j=j', k=k'
    
    For the Strassen decomposition of 2x2*2x2, we need to map the indices correctly.
    
    Let me build this more carefully.
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen formula for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e+f)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+d)(h-g)
    # M6 = (b-c)(e+g)
    # M7 = (b+d)(f-h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For a 2x2 * 2x2 block at position (i0, j0) in A and (j0, k0) in B:
    # A indices: (i0+di, j0+dj) for di,dj in {0,1}
    # B indices: (j0+dj, k0+dk) for dj,dk in {0,1}
    # C indices: (i0+di, k0+dk) for di,dk in {0,1}
    
    def add_strassen_block(i0, j0, k0, offset):
        """Add Strassen terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        # M1 = (a+b)(e+f)
        # a = A[i0, j0], b = A[i0, j0+1]
        # e = B[j0, k0], f = B[j0+1, k0]
        # Wait, B[j0, k0] means row j0, col k0.
        # In V, index is j'*p + k, so B[j0, k0] is at V[j0*p + k0]
        # In U, index is i*m + j, so A[i0, j0] is at U[i0*m + j0]
        # In W, index is k'*n + i', so C[k0, i0] is at W[k0*n + i0]
        
        # Let me re-examine. C[i, k] = sum_j A[i, j] B[j, k]
        # So C[i0+di, k0+dk] = sum_{dj} A[i0+di, j0+dj] B[j0+dj, k0+dk]
        
        # Strassen for the 2x2 block:
        # A_block = [[A[i0, j0], A[i0, j0+1]],
        #            [A[i0+1, j0], A[i0+1, j0+1]]]
        # B_block = [[B[j0, k0], B[j0, k0+1]],
        #            [B[j0+1, k0], B[j0+1, k0+1]]]
        
        # M1 = (A[i0,j0] + A[i0,j0+1]) * (B[j0,k0] + B[j0+1,k0])
        # This contributes to C[i0, k0] and C[i0, k0+1] and C[i0+1, k0] and C[i0+1, k0+1]
        
        # Actually, Strassen computes C = A*B where all are 2x2.
        # The 7 multiplications give us the 4 entries of C.
        
        # For M1: factor on A side: A[i0,j0] + A[i0,j0+1]
        #   In U: U[i0*m + j0, r] += 1, U[i0*m + j0+1, r] += 1
        #   In V: V[j0*p + k0, r] += 1, V[j0+1*p + k0, r] += 1
        #   Wait, this doesn't work directly because Strassen has specific combinations.
        
        # Let me think differently. The tensor entry T[a, b, c] = 1 iff the indices match.
        # For the schoolbook term for (i, j, k): U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
        # This gives T[i*m+j, j*p+k, k*n+i] += 1, which is correct when i=i', j=j', k=k'.
        
        # For Strassen, we need to find 7 terms that sum to the same tensor.
        # The 2x2*2x2 tensor has 8 non-zero entries (all combinations of di,dj,dk where the 
        # "matching" condition holds). Wait no, it has 4 non-zero entries:
        # T[i0*m+j0, j0*p+k0, k0*n+i0] = 1
        # T[i0*m+j0+1, (j0+1)*p+k0, k0*n+i0] = 1
        # T[(i0+1)*m+j0, j0*p+k0, k0*n+(i0+1)] = 1
        # T[(i0+1)*m+j0+1, (j0+1)*p+k0, k0*n+(i0+1)] = 1
        # Plus the cross terms... wait no.
        
        # Actually T[a, b, c] = 1 iff i=i', j=j', k=k'. So for the 2x2*2x2 block:
        # Only 4 entries are 1: (i0,j0,k0), (i0,j0+1,k0+1)... no wait.
        # i=i', j=j', k=k'. So:
        # (i0, j0, k0): 1
        # (i0, j0, k0+1): 0 (j=j' but k≠k')
        # (i0, j0+1, k0): 0
        # (i0, j0+1, k0+1): 0
        # (i0+1, j0, k0): 0
        # (i0+1, j0, k0+1): 0
        # (i0+1, j0+1, k0): 0
        # (i0+1, j0+1, k0+1): 1
        
        # Wait, that's only 2 entries! That can't be right for matrix multiplication.
        
        # Oh I see my confusion. The tensor T is defined as:
        # T[a, b, c] where a=(i,j), b=(j',k), c=(k',i')
        # T[a,b,c] = 1 iff i=i' AND j=j' AND k=k'
        
        # So for the full 2x4x5 problem, T has exactly 40 non-zero entries (one for each 
        # (i,j,k) triple), each equal to 1.
        
        # The schoolbook decomposition uses one rank-1 term per (i,j,k) triple.
        
        # For Strassen on a 2x2*2x2 sub-block, we're decomposing the 8 non-zero entries
        # of that sub-block (for i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}) into 7 terms.
        
        # The 8 entries are:
        # (i0, j0, k0), (i0, j0, k0+1), (i0, j0+1, k0), (i0, j0+1, k0+1)
        # (i0+1, j0, k0), (i0+1, j0, k0+1), (i0+1, j0+1, k0), (i0+1, j0+1, k0+1)
        
        # Each has T value 1.
        
        # Strassen's 7 terms for 2x2*2x2:
        # M1 = (a+b)(e+f) where a=A[i0,j0], b=A[i0,j0+1], e=B[j0,k0], f=B[j0+1,k0]
        #   U side: A[i0,j0] + A[i0,j0+1] -> U[i0*m+j0, r]=1, U[i0*m+j0+1, r]=1
        #   V side: B[j0,k0] + B[j0+1,k0] -> V[j0*p+k0, r]=1, V[j0+1*p+k0, r]=1
        #   W side: This contributes to C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
        #   But Strassen M1 contributes to C11 and C22 (with other terms).
        
        # I think I need to be more careful. Let me look up the exact Strassen tensor decomposition.
        
        # The 2x2*2x2 matrix multiplication tensor can be written as a sum of 7 rank-1 tensors.
        # The indices are (i, j, k) for i,j,k in {0,1}.
        
        # Term 1: (i=0 or 1, j=0 or 1, k=0) with signs... 
        
        # Actually, let me just use the known Strassen decomposition.
        # For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[c11,c12],[c21,c22]]
        
        # M1 = (a+b)(e+f): contributes +1 to c11, -1 to c21, -1 to c12, +1 to c22... no.
        
        # Standard Strassen:
        # M1 = (a+b)(e+f)
        # M2 = (c+d)(e+f)  
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+d)(h-g)
        # M6 = (b-c)(e+g)
        # M7 = (b+d)(f-h)
        
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Now, in tensor form, each M corresponds to a rank-1 tensor.
        # M1 = (a+b)(e+f) means:
        #   A-side: a + b = A[0,0] + A[0,1]
        #   B-side: e + f = B[0,0] + B[0,1]  (note: f is B[0,1], h is B[1,1])
        #   Wait: B = [[e,f],[g,h]], so e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        #   C-side: M1 contributes to c11 with +1, c22 with +1 (from c11=M1+... and c22=M1-...)
        #   Actually M1 appears in c11 (+1), c21 (0), c12 (0), c22 (+1)
        #   Wait no: c11 = M1 + M4 - M5 + M7, c22 = M1 - M2 + M3 + M6
        #   So M1 contributes +1 to c11 and +1 to c22.
        
        # Hmm, but in the tensor decomposition, each rank-1 term is a product of three vectors.
        # The C-side vector for M1 should have 1s where M1 contributes.
        # M1 contributes to c11 (i=0,k=0) and c22 (i=1,k=1).
        # So W-side: W[k*n+i] for k=0,i=0 and k=1,i=1 -> W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
        
        # Let me define the 7 Strassen terms for a 2x2*2x2 block with local indices (di,dj,dk) in {0,1}:
        
        # Term 1: M1 = (A[0,0]+A[0,1])(B[0,0]+B[1,0])
        #   U: A[0,0]+A[0,1] -> local (0,0) and (0,1)
        #   V: B[0,0]+B[1,0] -> local (0,0) and (1,0)  [B row 0 col 0, B row 1 col 0]
        #   Wait, B[0,0] is row 0, col 0. B[1,0] is row 1, col 0.
        #   But f = B[0,1], so (e+f) = B[0,0]+B[0,1], not B[0,0]+B[1,0].
        #   
        #   Let me re-read: M1 = (a+b)(e+f) where a=A[0,0], b=A[0,1], e=B[0,0], f=B[0,1]
        #   So (e+f) = B[0,0] + B[0,1]
        #   V side: B[0,0] + B[0,1] -> local rows {0}, cols {0,1}
        #   
        #   C contributions: c11 += M1, c22 += M1
        #   W side: c11 is (i=0,k=0), c22 is (i=1,k=1)
        
        # OK this is getting complex. Let me just hardcode the Strassen tensor decomposition.
        
        # For local indices (i,j,k) in {0,1}^3, the 7 Strassen terms are:
        
        # I'll define them as (U_coeffs, V_coeffs, W_coeffs) where each is a dict from (local_idx) to coeff.
        
        strassen_terms = [
            # M1: (a+b)(e+f), contributes to c11, c22
            # U: (0,0):1, (0,1):1
            # V: (0,0):1, (0,1):1  [B[0,0]+B[0,1]]
            # W: (0,0):1, (1,1):1  [c11, c22]
            ({(0,0):1, (0,1):1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
            
            # M2: (c+d)(e+f), contributes to c21, -c22
            # U: (1,0):1, (1,1):1
            # V: (0,0):1, (0,1):1
            # W: (0,1):1, (1,1):-1  [c21, c22]
            ({(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(0,1):1, (1,1):-1}),
            
            # M3: a(f-h), contributes to c12, c22
            # U: (0,0):1
            # V: (0,1):1, (1,1):-1  [B[0,1]-B[1,1]]
            # W: (0,1):1, (1,1):1  [c12, c22]
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            
            # M4: d(g-e), contributes to c11, c21
            # U: (1,1):1
            # V: (1,0):1, (0,0):-1  [B[1,0]-B[0,0]]
            # W: (0,0):1, (0,1):1  [c11, c21]
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            
            # M5: (a+d)(h-g), contributes to -c11, c12
            # U: (0,0):1, (1,1):1
            # V: (1,1):1, (1,0):-1  [B[1,1]-B[1,0]]
            # W: (0,0):-1, (0,1):1  [c11, c12]
            ({(0,0):1, (1,1):1}, {(1,1):1, (1,0):-1}, {(0,0):-1, (0,1):1}),
            
            # M6: (b-c)(e+g), contributes to c22
            # U: (0,1):1, (1,0):-1
            # V: (0,0):1, (1,0):1  [B[0,0]+B[1,0]]
            # W: (1,1):1
            ({(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
            
            # M7: (b+d)(f-h), contributes to c11, c12
            # U: (0,1):1, (1,1):1
            # V: (0,1):1, (1,1):-1  [B[0,1]-B[1,1]]
            # W: (0,0):1, (0,1):1  [c11, c12]
            ({(0,1):1, (1,1):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (0,1):1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for (di, dj), u_val in u_coeffs.items():
                U[(i0 + di) * m + (j0 + dj), offset] += u_val
            for (dj, dk), v_val in v_coeffs.items():
                V[(j0 + dj) * p + (k0 + dk), offset] += v_val
            for (dk, di), w_val in w_coeffs.items():
                W[(k0 + dk) * n + (i0 + di), offset] += w_val
            offset += 1
        
        return offset
    
    def add_schoolbook_2x2_2x1(i0, j0, k0, offset):
        """Add schoolbook terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        for di in range(2):
            for dj in range(2):
                for dk in range(1):
                    U[(i0 + di) * m + (j0 + dj), offset] = 1.0
                    V[(j0 + dj) * p + (k0 + dk), offset] = 1.0
                    W[(k0 + dk) * n + (i0 + di), offset] = 1.0
                    offset += 1
        return offset
    
    # Now build the full decomposition
    # A is 2x4, B is 4x5
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # Split B into B_top (2x5, rows 0-1) and B_bot (2x5, rows 2-3)
    # C = A_left * B_top + A_right * B_bot
    
    # For A_left * B_top: (2x2) * (2x5)
    # Split B_top into 3 column groups: cols 0-1, 2-3, 4
    # A_left * B_top[:, 0:2]: 2x2 * 2x2 -> Strassen (7 terms)
    # A_left * B_top[:, 2:4]: 2x2 * 2x2 -> Strassen (7 terms)
    # A_left * B_top[:, 4:5]: 2x2 * 2x1 -> schoolbook (4 terms)
    
    # For A_right * B_bot: (2x2) * (2x5)
    # Split B_bot into 3 column groups: cols 0-1, 2-3, 4
    # A_right * B_bot[:, 0:2]: 2x2 * 2x2 -> Strassen (7 terms)
    # A_right * B_bot[:, 2:4]: 2x2 * 2x2 -> Strassen (7 terms)
    # A_right * B_bot[:, 4:5]: 2x2 * 2x1 -> schoolbook (4 terms)
    
    offset = 0
    # A_left * B_top
    offset = add_strassen_block(0, 0, 0, offset)   # A[0:2, 0:2] * B[0:2, 0:2]
    offset = add_strassen_block(0, 0, 2, offset)   # A[0:2, 0:2] * B[0:2, 2:4]
    offset = add_schoolbook_2x2_2x1(0, 0, 4, offset)  # A[0:2, 0:2] * B[0:2, 4:5]
    
    # A_right * B_bot
    offset = add_strassen_block(0, 2, 0, offset)   # A[0:2, 2:4] * B[2:4, 0:2]
    offset = add_strassen_block(0, 2, 2, offset)   # A[0:2, 2:4] * B[2:4, 2:4]
    offset = add_schoolbook_2x2_2x1(0, 2, 4, offset)  # A[0:2, 2:4] * B[2:4, 4:5]
    
    assert offset == total_rank, f"Expected rank 36, got {offset}"
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    U_arr = U  # shape (n*m, rank)
    V_arr = V  # shape (m*p, rank)
    W_arr = W  # shape (p*n, rank)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = float(np.einsum('r,r,r->', U_arr[a], V_arr[b], W_arr[c]))
                            err = abs(actual - expected)
                            if err > max_err:
                                max_err = err
    return max_err

def _check_column_redundant(U, V, W, col_idx, tol=1e-9):
    """
    Check if column col_idx is linearly redundant, i.e., there exist coefficients
    alpha_r for r != col_idx such that:
    U[:, col_idx] = sum_{r != col_idx} alpha_r * U[:, r]
    V[:, col_idx] = sum_{r != col_idx} alpha_r * V[:, r]
    W[:, col_idx] = sum_{r != col_idx} alpha_r * W[:, r]
    
    If so, we can remove column col_idx and adjust the other columns.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, None
    
    # Build the system: find alpha such that
    # [U[:, r] for r != col_idx] * alpha = U[:, col_idx]
    # [V[:, r] for r != col_idx] * alpha = V[:, col_idx]
    # [W[:, r] for r != col_idx] * alpha = W[:, col_idx]
    
    other_cols = [r for r in range(rank) if r != col_idx]
    
    # Stack U, V, W vertically
    A_stack = np.vstack([U[:, other_cols], V[:, other_cols], W[:, other_cols]])  # shape (nU+nV+nW, rank-1)
    b_stack = np.concatenate([U[:, col_idx], V[:, col_idx], W[:, col_idx]])  # shape (nU+nV+nW,)
    
    # Solve least squares
    try:
        alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_stack, b_stack, rcond=tol)
    except np.linalg.LinAlgError:
        return False, None
    
    # Check if the solution is good
    recon = A_stack @ alpha
    err = np.linalg.norm(recon - b_stack)
    
    if err < tol * np.linalg.norm(b_stack) * 10:
        return True, alpha
    return False, None

def _remove_redundant_columns(U, V, W, max_iterations=50):
    """
    Iteratively remove redundant columns.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    removed = []
    for iteration in range(max_iterations):
        rank = U.shape[1]
        if rank <= 1:
            break
        
        found = False
        for col in range(rank):
            is_redundant, alpha = _check_column_redundant(U, V, W, col)
            if is_redundant:
                # Remove column col and adjust others
                other_cols = [r for r in range(rank) if r != col]
                for i, r in enumerate(other_cols):
                    U[:, r] += alpha[i] * U[:, col]
                    V[:, r] += alpha[i] * V[:, col]
                    W[:, r] += alpha[i] * W[:, col]
                
                # Remove the column
                U = np.delete(U, col, axis=1)
                V = np.delete(V, col, axis=1)
                W = np.delete(W, col, axis=1)
                
                removed.append(col)
                found = True
                break
        
        if not found:
            break
    
    return U, V, W, removed

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    # Build block-Strassen decomposition
    U, V, W = _build_block_strassen(n, m, p)
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        err = _verify_decomposition(U, V, W, n, m, p)
        if err > 1e-9:
            # Fall back to schoolbook
            U, V, W = _build_schoolbook(n, m, p)
    
    # Try to remove redundant columns
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        U, V, W, removed = _remove_redundant_columns(U, V, W)
        
        # Verify after removal
        if removed:
            err = _verify_decomposition(U, V, W, n, m, p)
            if err > 1e-9:
                # Rebuild and don't remove
                U, V, W = _build_block_strassen(n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Convert to int where possible
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U_list]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V_list]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
