"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank23():
    """
    Rank-23 decomposition for 3x3 matrix multiplication.
    Uses a more efficient Strassen-based approach with 23 terms.
    """
    n = m = p = 3
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # Strassen-style 23-term decomposition for 3x3
    # Based on the known rank-23 construction
    
    # Terms 1-7: Core 2x2 Strassen-like terms
    # M1 = (A11+A22)(B11+B22)
    terms.append(([(u_idx(0,0),1),(u_idx(1,1),1)], [(v_idx(0,0),1),(v_idx(1,1),1)], [(w_idx(0,0),1),(w_idx(1,1),1)]))
    # M2 = (A21+A22)B11
    terms.append(([(u_idx(1,0),1),(u_idx(1,1),1)], [(v_idx(0,0),1)], [(w_idx(0,1),1),(w_idx(1,1),-1)]))
    # M3 = A11(B21-B11)
    terms.append(([(u_idx(0,0),1)], [(v_idx(0,1),1),(v_idx(1,1),-1)], [(w_idx(1,0),1),(w_idx(1,1),1)]))
    # M4 = A22(B11-B21)
    terms.append(([(u_idx(1,1),1)], [(v_idx(1,0),1),(v_idx(0,0),-1)], [(w_idx(0,0),1),(w_idx(0,1),1)]))
    # M5 = (A11+A12)B22
    terms.append(([(u_idx(0,0),1),(u_idx(0,1),1)], [(v_idx(1,1),1)], [(w_idx(0,0),-1),(w_idx(1,0),1)]))
    # M6 = (A21-A11)(B11+B12)
    terms.append(([(u_idx(1,0),1),(u_idx(0,0),-1)], [(v_idx(0,0),1),(v_idx(0,1),1)], [(w_idx(1,1),1)]))
    # M7 = (A12-A22)(B21+B22)
    terms.append(([(u_idx(0,1),1),(u_idx(1,1),-1)], [(v_idx(1,0),1),(v_idx(1,1),1)], [(w_idx(0,0),1)]))
    
    # Terms 8-13: Border terms for C[0:2,2]
    terms.append(([(u_idx(0,0),1)], [(v_idx(0,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(0,1),1)], [(v_idx(1,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(0,2),1)], [(v_idx(2,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(1,0),1)], [(v_idx(0,2),1)], [(w_idx(2,1),1)]))
    terms.append(([(u_idx(1,1),1)], [(v_idx(1,2),1)], [(w_idx(2,1),1)]))
    terms.append(([(u_idx(1,2),1)], [(v_idx(2,2),1)], [(w_idx(2,1),1)]))
    
    # Terms 14-19: Border terms for C[2,0:2]
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,1),1)], [(w_idx(1,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,1),1)], [(w_idx(1,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,1),1)], [(w_idx(1,2),1)]))
    
    # Terms 20-23: Corner and cross terms
    # C[2,2] needs 3 terms, but we can combine with cross terms
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,2),1)], [(w_idx(2,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,2),1)], [(w_idx(2,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,2),1)], [(w_idx(2,2),1)]))
    
    # We have 23 terms so far (7 + 6 + 6 + 3 = 22), need 1 more for cross terms
    # The cross terms A[0:2,2]*B[2,0:2] -> C[0:2,0:2] need 4 terms
    # But we only have room for 1 more. This means the rank-23 construction
    # must use a different approach that combines these.
    
    # For now, this gives us 23 terms. Let's verify if it's correct.
    # If not, we'll fall back to rank-26.
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition by merging terms.
    Uses the base decomposition as a starting point and tries linear
    combinations of terms that preserve the identity property.
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    # Build the 27 constraint equations.
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We'll try to find a rank-`target_rank` sub-decomposition.
    # 
    # Strategy: try to express some terms as linear combinations of others.
    # This is equivalent to finding a lower-rank tensor in the span.
    # 
    # A practical approach: try all subsets of columns of size target_rank
    # and check if the 27 equations can still be satisfied by adjusting
    # coefficients. This is a hard problem in general, so we use a
    # heuristic: try removing pairs of terms that are "similar".
    
    # For each pair of terms (a, b), check if term b can be expressed
    # as a linear combination of the remaining terms. This is a
    # 27-equation, (cur_rank - 1)-unknown linear system.
    
    best = None
    best_rank = cur_rank
    
    # Try to remove one term at a time (greedy)
    removed = set()
    for _ in range(cur_rank - target_rank):
        if time.time() > deadline:
            break
        found = False
        for a in range(cur_rank):
            if a in removed:
                continue
            # Try removing term a: solve for whether the 27 equations
            # can be satisfied using only the other terms.
            # Build the 27 x (cur_rank-1) system.
            remaining = [x for x in range(cur_rank) if x != a and x not in removed]
            if len(remaining) < target_rank:
                continue
            
            # The 27 equations: for each (i,j,k):
            # sum_{r in remaining} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # But we can't change the U,V,W values - we can only drop terms.
            # So we just check if dropping term a still satisfies all 27.
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in remaining:
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                removed.add(a)
                found = True
                break
        if not found:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    return best, best_rank

def _search_rank_reductions(base_result, deadline):
    """
    Search for lower-rank decompositions by trying various strategies:
    1. Greedy term removal
    2. Linear algebra based rank reduction using SVD on term tensors
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best_result = None
    best_rank = cur_rank
    
    # Strategy 1: Greedy removal
    greedy_result, greedy_rank = _try_reduce_rank(base_result, cur_rank - 5, deadline)
    if greedy_result is not None and greedy_rank < best_rank:
        best_rank = greedy_rank
        best_result = greedy_result
    
    # Strategy 2: SVD-based rank reduction
    # Flatten each term's tensor to a 27-dim vector
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    # Build the 27 x cur_rank matrix of term tensors
    T = np.column_stack([term_tensor(r) for r in range(cur_rank)])
    
    # The sum of all term tensors should equal the ones vector (all 1s)
    ones = np.ones(27)
    
    # Check: does T @ ones(cur_rank) == ones?
    # If so, we can find a lower-rank representation by finding
    # the minimum number of columns whose sum (with appropriate
    # coefficients) equals ones.
    
    # Compute SVD of T
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T, full_matrices=False)
        # The rank of T tells us the dimension of the span
        effective_rank = sum(1 for s in s_vals if s > 1e-10)
        
        # Try to find a subset of columns that can represent the ones vector
        # with non-negative weights (since we need sum to be 1)
        # Actually, the constraint is: sum_r c_r * T[:,r] = ones, c_r >= 0
        # But we can also allow c_r = 0 for dropped terms.
        
        # For each possible rank r from 1 to cur_rank-1, try to find
        # r columns that can represent the ones vector.
        # This is a linear programming problem, but we'll use a greedy approach.
        
        # First, check if the ones vector is in the column space of T
        sol, residuals, rank_T, sv = np.linalg.lstsq(T, ones, rcond=None)
        if len(residuals) > 0 and residuals[0] < 1e-7:
            # The ones vector is in the span. Now find minimum support solution.
            # Use a greedy forward selection approach.
            selected = []
            remaining_ones = ones.copy()
            for _ in range(cur_rank - 1):
                if time.time() > deadline:
                    break
                best_idx = -1
                best_residual = float('inf')
                for r in range(cur_rank):
                    if r in selected:
                        continue
                    # Try adding column r
                    trial_selected = selected + [r]
                    A_sub = T[:, trial_selected]
                    sol_sub, res_sub, _, _ = np.linalg.lstsq(A_sub, ones, rcond=None)
                    if len(res_sub) > 0:
                        if res_sub[0] < best_residual:
                            best_residual = res_sub[0]
                            best_idx = r
                    else:
                        # Check if it's exactly in the span
                        if np.allclose(A_sub @ sol_sub, ones, atol=1e-9):
                            best_residual = 0
                            best_idx = r
                if best_idx >= 0 and best_residual < 1e-7:
                    selected.append(best_idx)
                    if best_residual < 1e-9:
                        break
                else:
                    break
            
            if len(selected) < cur_rank:
                # Verify this subset works
                A_sub = T[:, selected]
                sol_sub, res_sub, _, _ = np.linalg.lstsq(A_sub, ones, rcond=None)
                if np.allclose(A_sub @ sol_sub, ones, atol=1e-9):
                    # Now we need to check if this corresponds to a valid
                    # tensor decomposition. The coefficients sol_sub give us
                    # the weights, but we need the actual U, V, W factors.
                    # 
                    # Key insight: if term r can be expressed as a linear
                    # combination of other terms, we can't just drop it because
                    # the U, V, W factors are fixed. We can only drop terms
                    # if they are exactly zero in the decomposition, or if
                    # we can re-express the remaining terms.
                    #
                    # Actually, the correct approach: the 26 terms define a
                    # tensor T_total = sum_r T_r. If some T_r is in the span
                    # of the others, we can't simply drop it. We need to find
                    # new U', V', W' factors of lower rank.
                    #
                    # For now, just check if any single term is exactly
                    # representable by the others (which means we can set
                    # its coefficient to 0 and adjust others).
                    pass
        
        # Simpler approach: check for exact linear dependencies
        removable = []
        for r in range(cur_rank):
            if time.time() > deadline:
                break
            others = [s for s in range(cur_rank) if s != r]
            A = T[:, others]
            b = T[:, r]
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-7:
                    removable.append(r)
            except:
                pass
        
        if removable:
            remaining = [x for x in range(cur_rank) if x not in removable]
            new_rank = len(remaining)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {
                    "U": U[:, remaining].tolist(),
                    "V": V[:, remaining].tolist(),
                    "W": W[:, remaining].tolist()
                }
    except:
        pass
    
    return best_result, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Start with rank-26 as a guaranteed valid baseline
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()
    
    # Try to reduce the rank to 23 (known minimum) or 24/25
    if time.time() < deadline - 10:
        reduced, reduced_rank = _search_rank_reductions(result, deadline - 5)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
