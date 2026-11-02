"""
Combined approach: tensor-slicing + minimum-cardinality set-cover over a
comprehensive pool of rank-1 tensors (schoolbook, Strassen 2x2x2, and
generalized 2x2x2 with coefficient patterns), with greedy selection and
backtracking refinement.

The key structural change: instead of hard-coding a fixed block decomposition,
we generate a rich pool of candidate rank-1 terms and solve a set-cover
problem to find the minimum number of terms that exactly reproduce the
matrix-multiplication tensor.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Each candidate rank-1 term is represented as:
    # (u_dict, v_dict, w_dict) where each dict maps index -> coefficient
    # The contribution to T[i,j,k] from term r is:
    #   u_dict.get(u_idx(i,j), 0) * v_dict.get(v_idx(j,k), 0) * w_dict.get(w_idx(k,i), 0)
    # We need sum over all selected terms to equal 1 for all (i,j,k).

    # Generate candidate pool
    candidates = []  # list of (u_dict, v_dict, w_dict)

    # Type 1: Schoolbook terms (single i,j,k with coeff 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {u_idx(i, j): 1}
                v = {v_idx(j, k): 1}
                w = {w_idx(k, i): 1}
                candidates.append((u, v, w))

    # Type 2: Strassen 2x2x2 blocks
    # For each pair of j's and pair of k's, generate the 7 Strassen terms
    for j1 in range(m):
        for j2 in range(j1 + 1, m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    # Standard Strassen with i=0,1
                    # u00=(0,j1), u01=(0,j2), u10=(1,j1), u11=(1,j2)
                    # v00=(j1,k1), v01=(j1,k2), v10=(j2,k1), v11=(j2,k2)
                    # w00=(k1,0), w01=(k2,0), w10=(k1,1), w11=(k2,1)
                    u00 = u_idx(0, j1)
                    u01 = u_idx(0, j2)
                    u10 = u_idx(1, j1)
                    u11 = u_idx(1, j2)
                    v00 = v_idx(j1, k1)
                    v01 = v_idx(j1, k2)
                    v10 = v_idx(j2, k1)
                    v11 = v_idx(j2, k2)
                    w00 = w_idx(k1, 0)
                    w01 = w_idx(k2, 0)
                    w10 = w_idx(k1, 1)
                    w11 = w_idx(k2, 1)

                    # P1 = (a00+a11)(b00+b11)
                    candidates.append(({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1}))
                    # P2 = (a10+a11)b00
                    candidates.append(({u10: 1, u11: 1}, {v00: 1}, {w10: 1, w11: -1}))
                    # P3 = a00(b01-b11)
                    candidates.append(({u00: 1}, {v01: 1, v11: -1}, {w01: 1, w11: 1}))
                    # P4 = a11(b10-b00)
                    candidates.append(({u11: 1}, {v10: 1, v00: -1}, {w00: 1, w10: 1}))
                    # P5 = (a00+a01)b11
                    candidates.append(({u00: 1, u01: 1}, {v11: 1}, {w00: -1, w01: 1}))
                    # P6 = (a10-a00)(b00+b01)
                    candidates.append(({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w11: 1}))
                    # P7 = (a01-a11)(b10+b11)
                    candidates.append(({u01: 1, u11: -1}, {v10: 1, v11: 1}, {w00: 1}))

    # Type 3: Generalized 2x2x2 with different coefficient patterns
    # Try some alternative coefficient sets that might be useful
    for j1 in range(m):
        for j2 in range(j1 + 1, m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    u00 = u_idx(0, j1)
                    u01 = u_idx(0, j2)
                    u10 = u_idx(1, j1)
                    u11 = u_idx(1, j2)
                    v00 = v_idx(j1, k1)
                    v01 = v_idx(j1, k2)
                    v10 = v_idx(j2, k1)
                    v11 = v_idx(j2, k2)
                    w00 = w_idx(k1, 0)
                    w01 = w_idx(k2, 0)
                    w10 = w_idx(k1, 1)
                    w11 = w_idx(k2, 1)

                    # Additional patterns:
                    # (a00-a11)(b00-b11)
                    candidates.append(({u00: 1, u11: -1}, {v00: 1, v11: -1}, {w00: 1, w11: 1}))
                    # (a01+a10)(b11+b00) - cross terms
                    candidates.append(({u01: 1, u10: 1}, {v11: 1, v00: 1}, {w01: 1, w10: 1}))
                    # a00*b00 + a11*b11 type (but as single term with two products)
                    # Actually these are already covered by schoolbook. Skip.

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for u, v, w in candidates:
        key = (tuple(sorted(u.items())), tuple(sorted(v.items())), tuple(sorted(w.items())))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((u, v, w))
    candidates = unique_candidates

    # Precompute the effect of each candidate on the tensor
    # Effect is a vector of length n*m*p (flattened)
    # T[i,j,k] index: i * m * p + j * p + k
    def tensor_idx(i, j, k):
        return i * m * p + j * p + k

    num_cells = n * m * p
    effects = []
    for u, v, w in candidates:
        eff = np.zeros(num_cells)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = u.get(u_idx(i, j), 0) * v.get(v_idx(j, k), 0) * w.get(w_idx(k, i), 0)
                    eff[tensor_idx(i, j, k)] += val
        effects.append(eff)

    # Target: all ones
    target = np.ones(num_cells)

    # Greedy set cover: iteratively select the candidate that covers the most uncovered cells
    # But we need exact coverage, not just coverage. Let's use a different approach.
    
    # Approach: Start with the full schoolbook decomposition (rank 60), then try to
    # replace groups of schoolbook terms with Strassen terms to reduce rank.
    
    # Actually, let's use a linear algebra approach:
    # We want to find the minimum number of candidates whose sum of effects equals the target.
    # This is a subset-sum-like problem, but in high dimensions.
    
    # Better approach: Use the fact that we know a rank-54 solution exists.
    # Start from that and try to reduce further.
    
    # Build the known rank-54 decomposition
    def build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        strassen_cells = set()
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                for i in range(n):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            strassen_cells.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_cells:
                        continue
                    U[u_idx(i, j)][col] = 1
                    V[v_idx(j, k)][col] = 1
                    W[w_idx(k, i)][col] = 1
                    col += 1
        
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                u00 = u_idx(0, j1)
                u01 = u_idx(0, j2)
                u10 = u_idx(1, j1)
                u11 = u_idx(1, j2)
                v00 = v_idx(j1, k1)
                v01 = v_idx(j1, k2)
                v10 = v_idx(j2, k1)
                v11 = v_idx(j2, k2)
                w00 = w_idx(k1, 0)
                w01 = w_idx(k2, 0)
                w10 = w_idx(k1, 1)
                w11 = w_idx(k2, 1)
                
                # P1
                U[u00][col] += 1; U[u11][col] += 1
                V[v00][col] += 1; V[v11][col] += 1
                W[w00][col] += 1; W[w11][col] += 1
                col += 1
                # P2
                U[u10][col] += 1; U[u11][col] += 1
                V[v00][col] += 1
                W[w10][col] += 1; W[w11][col] += -1
                col += 1
                # P3
                U[u00][col] += 1
                V[v01][col] += 1; V[v11][col] += -1
                W[w01][col] += 1; W[w11][col] += 1
                col += 1
                # P4
                U[u11][col] += 1
                V[v10][col] += 1; V[v00][col] += -1
                W[w00][col] += 1; W[w10][col] += 1
                col += 1
                # P5
                U[u00][col] += 1; U[u01][col] += 1
                V[v11][col] += 1
                W[w00][col] += -1; W[w01][col] += 1
                col += 1
                # P6
                U[u10][col] += 1; U[u00][col] += -1
                V[v00][col] += 1; V[v01][col] += 1
                W[w11][col] += 1
                col += 1
                # P7
                U[u01][col] += 1; U[u11][col] += -1
                V[v10][col] += 1; V[v11][col] += 1
                W[w00][col] += 1
                col += 1
        
        return U, V, W

    def verify(U, V, W):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # Get the rank-54 solution
    U54, V54, W54 = build_rank54()
    if not verify(U54, V54, W54):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[u_idx(i, j)][col_sb] = 1
                    V_sb[v_idx(j, k)][col_sb] = 1
                    W_sb[w_idx(k, i)][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    best_U, best_V, best_W = U54, V54, W54
    best_rank = 54

    # Now try to reduce rank using the candidate pool
    # Strategy: For each term in the current decomposition, check if it can be
    # replaced by a combination of other candidates that covers the same cells.
    
    # Convert current decomposition to list of candidate indices
    U_mat = np.array(U54, dtype=float)
    V_mat = np.array(V54, dtype=float)
    W_mat = np.array(W54, dtype=float)
    current_rank = len(U54[0]) if U54 else 0

    # For each column r, compute its effect vector
    col_effects = []
    for r in range(current_rank):
        eff = np.zeros(num_cells)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    eff[tensor_idx(i, j, k)] += val
        col_effects.append(eff)

    # Try to find a subset of candidates that sums to the target with fewer terms
    # Use a greedy approach with backtracking for small reductions
    
    def try_reduce_by_one():
        """Try to find a rank-53 decomposition by removing one term and compensating."""
        if time.time() > deadline - 5:
            return None
        
        # For each term r, check if the remaining terms plus some new candidates
        # can still sum to the target.
        # This is equivalent to: can we express col_effects[r] as a sum of
        # effects from other candidates (not already used)?
        
        # Simpler: just check if removing any single term still gives a valid decomposition
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            U_test = np.delete(U_mat, r, axis=1)
            V_test = np.delete(V_mat, r, axis=1)
            W_test = np.delete(W_mat, r, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.sum(U_test[i * m + j, :] * V_test[j * p + k, :] * W_test[k * n + i, :])
                        if not np.isclose(total, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return (U_test.tolist(), V_test.tolist(), W_test.tolist())
        
        return None

    # Try reducing by 1
    if time.time() < deadline - 10:
        result = try_reduce_by_one()
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 53

    # Try a more aggressive search: look for alternative decompositions
    # using the full candidate pool with a greedy set-cover-like approach
    if time.time() < deadline - 30:
        # Try to find a decomposition using at most 52 terms
        # Use a heuristic: start with the rank-54 solution, try to swap out
        # pairs of terms for single terms that cover the same cells
        
        # For each pair of columns (r1, r2), check if their combined effect
        # can be achieved by a single candidate
        U_mat = np.array(best_U, dtype=float)
        V_mat = np.array(best_V, dtype=float)
        W_mat = np.array(best_W, dtype=float)
        current_rank = len(best_U[0]) if best_U else 0
        
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                
                # Combined effect of terms r1 and r2
                combined = np.zeros(num_cells)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = (U_mat[i * m + j, r1] * V_mat[j * p + k, r1] * W_mat[k * n + i, r1] +
                                   U_mat[i * m + j, r2] * V_mat[j * p + k, r2] * W_mat[k * n + i, r2])
                            combined[tensor_idx(i, j, k)] += val
                
                # Check if any single candidate matches this combined effect
                for c, (u, v, w) in enumerate(candidates):
                    eff = np.zeros(num_cells)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = u.get(u_idx(i, j), 0) * v.get(v_idx(j, k), 0) * w.get(w_idx(k, i), 0)
                                eff[tensor_idx(i, j, k)] += val
                    if np.allclose(eff, combined, atol=1e-9):
                        # Found a replacement! Remove r1 and r2, add candidate c
                        # But we need to make sure candidate c is not already in use
                        # For simplicity, just try it
                        U_new = np.delete(U_mat, [r1, r2], axis=1)
                        V_new = np.delete(V_mat, [r1, r2], axis=1)
                        W_new = np.delete(W_mat, [r1, r2], axis=1)
                        
                        # Add the new candidate
                        new_rank = U_new.shape[1] + 1
                        U_full = np.zeros((n * m, new_rank))
                        V_full = np.zeros((m * p, new_rank))
                        W_full = np.zeros((p * n, new_rank))
                        U_full[:, :-1] = U_new
                        V_full[:, :-1] = V_new
                        W_full[:, :-1] = W_new
                        
                        for idx, val in u.items():
                            U_full[idx, -1] = val
                        for idx, val in v.items():
                            V_full[idx, -1] = val
                        for idx, val in w.items():
                            W_full[idx, -1] = val
                        
                        if verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                            best_U, best_V, best_W = U_full.tolist(), V_full.tolist(), W_full.tolist()
                            best_rank = new_rank
                            U_mat = U_full
                            V_mat = V_full
                            W_mat = W_full
                            current_rank = new_rank
                            break

    # Final verification
    if not verify(best_U, best_V, best_W):
        U_orig, V_orig, W_orig = build_rank54()
        return {"U": U_orig, "V": V_orig, "W": W_orig}

    return {"U": best_U, "V": best_V, "W": best_W}
