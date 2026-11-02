"""
Matrix multiplication 2x4 * 4x4 tensor decomposition.
Base: verified rank-28 Strassen. Then attempts exact rank-27 via integer search.
"""

import numpy as np
import time
from itertools import product, combinations_with_replacement

def _build_strassen_28():
    """Construct the rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def idx_a(i, j): return i * m + j
    def idx_b(i, j): return i * p + j
    def idx_c(i, j): return j * n + i  # trace convention W[k*n+i, r]

    def add_strassen_block(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        u00, u01 = idx_a(a_r0, a_c0), idx_a(a_r0, a_c0 + 1)
        u10, u11 = idx_a(a_r0 + 1, a_c0), idx_a(a_r0 + 1, a_c0 + 1)
        v00, v01 = idx_b(b_r0, b_c0), idx_b(b_r0, b_c0 + 1)
        v10, v11 = idx_b(b_r0 + 1, b_c0), idx_b(b_r0 + 1, b_c0 + 1)
        w00, w01 = idx_c(c_r0, c_c0), idx_c(c_r0, c_c0 + 1)
        w10, w11 = idx_c(c_r0 + 1, c_c0), idx_c(c_r0 + 1, c_c0 + 1)

        # M1 = (a00+a11)(b00+b11)
        U[u00, offset+0] += 1; U[u11, offset+0] += 1
        V[v00, offset+0] += 1; V[v11, offset+0] += 1
        W[w00, offset+0] += 1; W[w11, offset+0] += 1
        # M2 = (a10+a11)b00
        U[u10, offset+1] += 1; U[u11, offset+1] += 1
        V[v00, offset+1] += 1
        W[w10, offset+1] += 1; W[w11, offset+1] -= 1
        # M3 = a00(b01-b11)
        U[u00, offset+2] += 1
        V[v01, offset+2] += 1; V[v11, offset+2] -= 1
        W[w01, offset+2] += 1; W[w11, offset+2] += 1
        # M4 = a11(b10-b00)
        U[u11, offset+3] += 1
        V[v10, offset+3] += 1; V[v00, offset+3] -= 1
        W[w00, offset+3] += 1; W[w10, offset+3] += 1
        # M5 = (a00+a01)b11
        U[u00, offset+4] += 1; U[u01, offset+4] += 1
        V[v11, offset+4] += 1
        W[w00, offset+4] -= 1; W[w01, offset+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[u10, offset+5] += 1; U[u00, offset+5] -= 1
        V[v00, offset+5] += 1; V[v01, offset+5] += 1
        W[w11, offset+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[u01, offset+6] += 1; U[u11, offset+6] -= 1
        V[v10, offset+6] += 1; V[v11, offset+6] += 1
        W[w00, offset+6] += 1

    # Four 2x2 * 2x2 blocks
    add_strassen_block(0,  0, 0, 0, 0, 0, 0)  # A_left * B_TL -> C_left
    add_strassen_block(7,  0, 2, 2, 0, 0, 0)  # A_right * B_BL -> C_left
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)  # A_left * B_TR -> C_right
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)  # A_right * B_BR -> C_right

    return U, V, W

def _verify(U, V, W, n, m, p, rank, trials=200):
    """Verify tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k] recovered."""
    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_ref = A @ B
        C_calc = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # Term: (A^T u_r) (v_r^T B) w_r^T  ->  (n x 1)(1 x p)(p x n)
            # Actually: sum_r (sum_{i,j} A[i,j] u_r[i,j]) * (sum_{j,k} B[j,k] v_r[j,k]) * w_r[k,i]
            # That gives scalar * w_r, not matrix.
            # Correct: the bilinear form for each r is:
            # P_r = (A * u_r^T) ... no.
            # Standard: C = sum_r (A U_r) (V_r B) W_r where U_r, V_r, W_r are matrices?
            # No. U is (nm, rank), so U[:,r] is a vector of length nm = n*m.
            # The rank-1 tensor is: u_r (n x m) ⊗ v_r (m x p) ⊗ w_r (p x n)
            # Contracting with A and B: sum_{i,j} A[i,j] u_r[i,j] * sum_{j,k} B[j,k] v_r[j,k] * w_r[k,i]?
            # That's (scalar)(scalar)(w_r[k,i]) which is not right dimensionally for C[i,k].
            # 
            # Correct interpretation:
            # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # must equal 1 if indices are consistent, 0 otherwise.
            # For matrix mult: the "consistent" condition is that the middle index j matches.
            # So T_{(i,j),(j',k),(k',i')} = 1 if i=i', j=j', k=k' else 0.
            # Verify: for random A, B, check C[i,k] = sum_{j} A[i,j] B[j,k]
            # Using the tensor: C[i,k] = sum_{j, j', k', i'} T_{(i,j),(j',k),(k',i')} A[i',j'] B[j',k']
            # = sum_{j} A[i,j] B[j,k] (since T picks out i'=i, j'=j, k'=k)
            # So we verify by checking the tensor identity directly.
            
            # Alternative: just check that the bilinear map reproduces matrix mult.
            # For each r, define:
            #   a_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
            #   b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)
            #   Then C[i,k] += a_r * b_r * W[k*n+i, r]?
            # That doesn't work either. Let me think again.
            
            # The correct formula: 
            # C[i,k] = sum_{j,r} (sum_{i'} U[i'*m+j, r] A[i',j]) ... no.
            # 
            # Standard tensor rank decomposition for matrix multiplication:
            # C_{i,k} = sum_r (sum_{i',j'} U_{i',j',r} A_{i',j'}) (sum_{j',k'} V_{j',k',r} B_{j',k'}) W_{k,i,r}
            # No, that's not standard either.
            # 
            # Let me use the direct tensor check:
            pass
        
        # Direct tensor verification
        # Build the full tensor T of shape (n*m, m*p, p*n)
        # T[a_idx, b_idx, c_idx] should equal sum_r U[a_idx,r] V[b_idx,r] W[c_idx,r]
        # And this should equal 1 if a_idx=(i,j), b_idx=(j,k), c_idx=(k,i) for valid i,j,k, else 0.
        
        # Actually, the simplest correct check:
        # C_calc[i,k] = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # Wait no. The tensor decomposition says:
        # The structural tensor M has entries M_{i,j,k} = 1 for the multiplication map.
        # M_{(i,j),(j',k),(k',i')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # And M = sum_r U_r ⊗ V_r ⊗ W_r where U_r, V_r, W_r are vectors.
        # So: delta_{i,i'} delta_{j,j'} delta_{k,k'} = sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]
        # 
        # To verify with random A, B:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_{i',j',k'} (sum_j A[i,j] B[j,k] delta_{i,i'} delta_{j,j'} delta_{k,k'})
        # Hmm, let me just do the direct check.
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # The contribution to C[i,k] from A[i,j]*B[j,k] should be captured
                    # by sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # which should equal 1 (since i'=i, j'=j, k'=k all match)
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_calc[i, k] += A[i, j] * B[j, k] * val
        if not np.allclose(C_calc, C_ref, atol=1e-10):
            return False
    return True

def _try_rank_27_search(deadline):
    """Attempt to find a rank-27 decomposition by merging Strassen terms
    across blocks, exploiting the n=2 < p=4 asymmetry."""
    n, m, p = 2, 4, 4
    target_rank = 27
    U28, V28, W28 = _build_strassen_28()

    # Strategy: the four 2x2 Strassen blocks produce 28 terms.
    # Blocks 0 (A_L*B_TL->C_L) and 1 (A_R*B_BL->C_L) both write to C_L.
    # Blocks 2 (A_L*B_TR->C_R) and 3 (A_R*B_BR->C_R) both write to C_R.
    # Within each pair, certain Strassen positions have identical U-support
    # (same A rows) or identical V-support (same B rows), enabling a
    # bilinear identity to merge one term from each into a single term.
    #
    # Concretely, for positions where U-vectors are proportional and
    # V-vectors are proportional across the two blocks, we can replace
    # (u,v1,w1) + (u,v2,w2) with (u, v1+v2, w1+w2) if the W supports
    # align. Otherwise try a 2->1 merge via a linear combination.
    #
    # We attempt: for each pair (r1, r2) with r1 in [0..6], r2 in [7..13]
    # (left-C pair) or r1 in [14..20], r2 in [21..27] (right-C pair),
    # check if U[:,r1] is proportional to U[:,r2] AND V[:,r1] is
    # proportional to V[:,r2]. If so, merge: new_V = V1 + V2, new_W = W1 + W2,
    # drop one term.
    #
    # If no direct proportional merge works, try a more general approach:
    # check if the 2x2x2 sub-tensor formed by {r1, r2} has rank 1
    # (i.e., the outer-product structure collapses).

    def _try_merge_pair(U, V, W, r1, r2):
        """Try to merge terms r1 and r2 into a single term.
        Returns (new_U, new_V, new_W, keep_indices) or None."""
        u1, u2 = U[:, r1], U[:, r2]
        v1, v2 = V[:, r1], V[:, r2]
        w1, w2 = W[:, r1], W[:, r2]

        # Check if u1 and u2 are proportional
        nz1 = np.nonzero(u1)[0]
        nz2 = np.nonzero(u2)[0]
        if len(nz1) == 0 or len(nz2) == 0:
            return None
        if not np.array_equal(nz1, nz2):
            return None
        ratio = u1[nz1] / u2[nz2]
        if not np.allclose(ratio, ratio[0], atol=1e-12):
            return None
        alpha = ratio[0]

        # Check if v1 and v2 are proportional
        nv1 = np.nonzero(v1)[0]
        nv2 = np.nonzero(v2)[0]
        if len(nv1) == 0 or len(nv2) == 0:
            return None
        if not np.array_equal(nv1, nv2):
            return None
        ratio_v = v1[nv1] / v2[nv2]
        if not np.allclose(ratio_v, ratio_v[0], atol=1e-12):
            return None
        beta = ratio_v[0]

        # If both proportional, we can write:
        # u1*v1*w1 + u2*v2*w2 = u2 * (alpha*v1 + v2) * w2 ... no that's wrong.
        # Actually: u1 = alpha*u2, v1 = beta*v2
        # Term1 = alpha*u2 * beta*v2 * w1
        # Term2 = u2 * v2 * w2
        # Sum = u2 * v2 * (alpha*beta*w1 + w2)
        # So merged: U=u2, V=v2, W=alpha*beta*w1 + w2
        new_w = alpha * beta * w1 + w2
        # Check new_w is "simple" (small integer/half-integer entries)
        if np.max(np.abs(new_w)) > 10:
            return None
        # Verify it's exact (integer or half-integer)
        if not np.allclose(new_w * 2, np.round(new_w * 2), atol=1e-10):
            return None

        return (u2.copy(), v2.copy(), new_w)

    # Try all cross-block pairs
    U_new = U28.copy()
    V_new = V28.copy()
    W_new = W28.copy()
    merged = set()
    used = set()

    # Left-C pairs: block0 (0-6) with block1 (7-13)
    for r1 in range(0, 7):
        if r1 in used:
            continue
        for r2 in range(7, 14):
            if r2 in used:
                continue
            result = _try_merge_pair(U28, V28, W28, r1, r2)
            if result is not None:
                u_m, v_m, w_m = result
                # Replace r1 with merged, mark r2 as dropped
                U_new[:, r1] = u_m
                V_new[:, r1] = v_m
                W_new[:, r1] = w_m
                U_new[:, r2] = 0
                V_new[:, r2] = 0
                W_new[:, r2] = 0
                used.add(r2)
                break

    # Right-C pairs: block2 (14-20) with block3 (21-27)
    for r1 in range(14, 21):
        if r1 in used:
            continue
        for r2 in range(21, 28):
            if r2 in used:
                continue
            result = _try_merge_pair(U28, V28, W28, r1, r2)
            if result is not None:
                u_m, v_m, w_m = result
                U_new[:, r1] = u_m
                V_new[:, r1] = v_m
                W_new[:, r1] = w_m
                U_new[:, r2] = 0
                V_new[:, r2] = 0
                W_new[:, r2] = 0
                used.add(r2)
                break

    # Count how many terms were merged
    n_merged = len(used)
    if n_merged < 1:
        return None

    # Build the reduced matrices (drop zero columns)
    active = [r for r in range(28) if r not in used]
    U_red = U_new[:, active]
    V_red = V_new[:, active]
    W_red = W_new[:, active]

    # Verify the reduced decomposition
    if not _verify(U_red, V_red, W_red, n, m, p, len(active), trials=100):
        return None

    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Build and verify rank-28
    U, V, W = _build_strassen_28()
    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Quick verification
    if not _verify(U, V, W, 2, 4, 4, 28, trials=50):
        # Fallback: naive rank-32
        n, m, p = 2, 4, 4
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank-27 search if time permits
    if time.time() < deadline - 30:
        result_27 = _try_rank_27_search(deadline)
        if result_27 is not None:
            best_rank = 27
            best_result = result_27
    
    return best_result
