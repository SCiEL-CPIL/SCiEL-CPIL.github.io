import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    n, m, p = 4, 4, 6

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Build rank-77 Strassen decomposition
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen (Kronecker product of two 2x2 Strassen)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    # Check deadline
    if time.time() > deadline - 40.0:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Verify correctness
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                residual += (s - expected)**2
    
    if residual > 1e-10:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Attempt rank reduction by finding linear dependencies among rank-1 components
    # The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If we can find coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0,
    # we can reduce rank. This is equivalent to finding the null space of the
    # "tensor unfolding" matrix.
    
    # Build matrix M where M[(i,j,k), r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Then find null space of M^T (or use SVD to find low-rank structure)
    
    num_tensors = n * m * p  # 4*4*6 = 96
    M = np.zeros((num_tensors, total_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(total_rank):
                    M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                idx += 1
    
    # Check if columns of M are linearly dependent
    # Use SVD to find effective rank
    try:
        if time.time() < deadline - 30.0:
            U_sv, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-8 * s_vals[0] if len(s_vals) > 0 else 1e-8
            effective_rank = np.sum(s_vals > tol)
            
            if effective_rank < total_rank:
                # There are linear dependencies - try to exploit them
                # Find a basis for the null space of M^T
                # Columns of Vt corresponding to small singular values span the null space
                null_mask = s_vals <= tol
                if np.any(null_mask):
                    # The null space vectors give linear combinations of columns that sum to zero
                    # We can use this to reduce rank, but it's complex to implement safely
                    pass
    except:
        pass

    # Simpler approach: try removing each column and checking if the remaining
    # columns can still span the same tensor space (approximately)
    # For each column r, compute its "contribution vector" and check if it's
    # in the span of the others.
    
    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column one at a time
    for r_remove in range(total_rank):
        if time.time() > deadline - 15.0:
            break
        
        # Check if column r_remove is linearly dependent on others
        # by seeing if its contribution vector is in the span of the rest
        mask = np.ones(total_rank, dtype=bool)
        mask[r_remove] = False
        remaining_cols = np.where(mask)[0]
        
        if len(remaining_cols) == 0:
            continue
        
        # Build the contribution matrix for remaining columns
        M_rem = M[:, remaining_cols]
        
        # Check if M[:, r_remove] is in the column space of M_rem
        # Use least squares
        try:
            coeffs, residuals, rank_rem, _ = np.linalg.lstsq(M_rem, M[:, r_remove], rcond=1e-10)
            recon = M_rem @ coeffs
            err = np.linalg.norm(M[:, r_remove] - recon)
            
            if err < 1e-6 * max(1.0, np.linalg.norm(M[:, r_remove])):
                # Column r_remove is (approximately) in the span of others
                # Remove it
                new_U = U[:, mask]
                new_V = V[:, mask]
                new_W = W[:, mask]
                
                # Verify the new decomposition still works
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(len(remaining_cols)):
                                s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                            res += (s - 1.0)**2
                
                if res < 1e-8:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(remaining_cols)
                    # Update M for subsequent iterations
                    M = M[:, mask]
                    total_rank = best_rank
                    # Restart the loop with reduced M
                    break
        except:
            pass
    
    if best_rank < total_rank:
        U, V, W = best_U, best_V, best_W
        total_rank = best_rank

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
