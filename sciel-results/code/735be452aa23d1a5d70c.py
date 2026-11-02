"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Start from a known valid decomposition (schoolbook rank 216), then attempt
rank reduction by merging columns where U and V vectors are linearly dependent.
We use a greedy approach: try to combine pairs of columns that have proportional U and V vectors.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses a model-and-optimize approach: parameterize the decomposition and attempt
    to reduce rank by finding linear dependencies among the rank-1 components.
    """
    deadline = time.time() + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 216)
    # Each term: U[row, col] = 1 if row corresponds to (i,j), V[col, col] = 1 if col corresponds to (j,k)
    # W[col, col] = 1 if it maps to output (k,i)
    
    rank_schoolbook = n * m * p  # 216
    
    # Build schoolbook factors
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                # W index: W[k*n+i, r]
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify correctness of schoolbook
    # C = A @ B, where (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For a single column corresponding to (i',j',k'), this contributes:
    #   1 * 1 * 1 to position (k',i') in W, which maps to output[i',k']
    # So summing over all columns: output[i,k] = sum_{j} A[i,j]*B[j,k] * (sum over matching i',j',k')
    # This is correct.
    
    # Now attempt rank reduction
    # Strategy: Find groups of columns where U vectors and V vectors are linearly dependent
    # If two columns have U[:,c1] = alpha * U[:,c2] and V[:,c1] = beta * V[:,c2],
    # then W[:,c1] must equal alpha*beta * W[:,c2] for them to be mergeable.
    # In that case, we can keep one column and scale W accordingly.
    
    # More generally, if a set of columns has U vectors spanning a d-dimensional space
    # and V vectors spanning a d-dimensional space, we can potentially reduce to d columns.
    
    # Let's use a greedy approach: repeatedly find pairs that can be merged.
    # Two columns c1, c2 can be merged if U[:,c1] and U[:,c2] are linearly dependent
    # AND V[:,c1] and V[:,c2] are linearly dependent.
    
    # For schoolbook, each column has U with a single 1 and V with a single 1.
    # No two columns have dependent U vectors (each has a different position with 1).
    # So schoolbook can't be reduced this way directly.
    
    # Alternative approach: Use block structure with Strassen at the 2x2 level.
    # 6 = 3 * 2, so we have a 3x3 block matrix with 2x2 blocks.
    # Use Strassen (rank 7) for each 2x2 multiplication.
    # Total rank: 9 * 3 * 7 = 189
    
    # But we can do better by using a smarter block decomposition.
    # Let's try 6 = 2 * 3, so 2x2 blocks with 3x3 submatrices? No, that's the same.
    
    # Actually, let's just implement the block-Strassen approach which gives rank 189.
    # Then try to reduce further.
    
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Now try to reduce rank from 189
    # Look for linear dependencies among columns
    # If columns c1..ck have U vectors in a space of dimension d < k,
    # and similarly for V, we might be able to reduce.
    
    # Use SVD-based approach: find the effective rank of U and V submatrices
    # for groups of columns, then attempt to compress.
    
    best_U, best_V, best_W = U_bs, V_bs, W_bs
    best_rank = rank_bs
    
    # Try to identify and merge groups of columns that are linearly dependent
    # For simplicity, let's check if any subset of columns can be compressed
    
    # Time check
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt: find the numerical rank of U_bs and V_bs
    # If the columns of U are linearly dependent, we can potentially reduce rank
    
    # Compute the rank of U_bs (should be at most n*m = 36)
    # and rank of V_bs (at most m*p = 36)
    # The tensor rank is at least max(rank(U), rank(V), rank(W))
    
    # Let's try a different approach: use the fact that the 6x6 multiplication
    # tensor has known lower bounds and try known algorithms.
    
    # For now, return the block-Strassen result (rank 189)
    # which is better than schoolbook (216)
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to further reduce by checking for mergeable columns
    # Two columns can be merged if U[:,c1] is proportional to U[:,c2]
    # and V[:,c1] is proportional to V[:,c2]
    
    # In our construction, each column has specific non-zero patterns.
    # Let's check for exact proportional relationships.
    
    reduced_U = [U_bs[:, c].copy() for c in range(rank_bs)]
    reduced_V = [V_bs[:, c].copy() for c in range(rank_bs)]
    reduced_W = [W_bs[:, c].copy() for c in range(rank_bs)]
    
    used = [True] * rank_bs
    merged = 0
    
    for i in range(rank_bs):
        if not used[i]:
            continue
        for j in range(i+1, rank_bs):
            if not used[j]:
                continue
            if time.time() > deadline - 10:
                break
            
            # Check if U[:,j] is proportional to U[:,i]
            ui = reduced_U[i]
            uj = reduced_U[j]
            
            # Find a non-zero element in ui to determine the ratio
            ratio_u = None
            for idx in range(len(ui)):
                if abs(ui[idx]) > 1e-12:
                    if abs(uj[idx]) > 1e-12:
                        if ratio_u is None:
                            ratio_u = uj[idx] / ui[idx]
                        elif abs(uj[idx] / ui[idx] - ratio_u) > 1e-9:
                            ratio_u = None
                        break
                    else:
                        ratio_u = None
                        break
            
            if ratio_u is None:
                continue
            
            # Check if V[:,j] is proportional to V[:,i] with some ratio
            vi = reduced_V[i]
            vj = reduced_V[j]
            
            ratio_v = None
            for idx in range(len(vi)):
                if abs(vi[idx]) > 1e-12:
                    if abs(vj[idx]) > 1e-12:
                        if ratio_v is None:
                            ratio_v = vj[idx] / vi[idx]
                        elif abs(vj[idx] / vi[idx] - ratio_v) > 1e-9:
                            ratio_v = None
                        break
                    else:
                        ratio_v = None
                        break
            
            if ratio_v is None:
                continue
            
            # Check if W[:,j] == ratio_u * ratio_v * W[:,i]
            wi = reduced_W[i]
            wj = reduced_W[j]
            expected_w = ratio_u * ratio_v * wi
            
            # Exact check for integer/half-integer entries
            # expected_w = ratio_u * ratio_v * wi
            # ratio_u and ratio_v are rational numbers (from integer entries)
            # We can check if wj == expected_w exactly
            # To avoid floating point issues, we can use fractions or just high precision
            # But since entries are small integers, float64 should be exact for products
            if np.all(wj == expected_w):
                # Merge j into i
                reduced_W[i] = wi + wj
                used[j] = False
                merged += 1
    
    if merged > 0:
        new_rank = sum(used)
        final_U = np.column_stack([reduced_U[i] for i in range(rank_bs) if used[i]])
        final_V = np.column_stack([reduced_V[i] for i in range(rank_bs) if used[i]])
        final_W = np.column_stack([reduced_W[i] for i in range(rank_bs) if used[i]])
        
        if new_rank < best_rank:
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
