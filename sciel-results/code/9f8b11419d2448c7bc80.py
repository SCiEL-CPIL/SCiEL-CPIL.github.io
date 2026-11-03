"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs):
    """Build a Strassen-based decomposition with given pairings."""
    rank = 7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _try_eliminate_terms(U, V, W, n, m, p, rank, max_attempts=200):
    """
    Try to eliminate redundant terms by checking if removing each term
    still gives a valid decomposition. Use iterative removal.
    """
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    cur_rank = rank
    
    eliminated = set()
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        found = False
        
        # Try terms in order of increasing "importance" (norm)
        norms = []
        for r in range(cur_rank):
            if r in eliminated:
                continue
            norm_sq = 0.0
            for i in range(n*m):
                norm_sq += U_cur[i][r]**2
            for l in range(m*p):
                norm_sq += V_cur[l][r]**2
            for k in range(p*n):
                norm_sq += W_cur[k][r]**2
            norms.append((norm_sq, r))
        
        norms.sort()
        
        for norm_sq, r in norms:
            # Try removing term r
            U_test = [row[:] for row in U_cur]
            V_test = [row[:] for row in V_cur]
            W_test = [row[:] for row in W_cur]
            
            for i in range(n*m):
                U_test[i][r] = 0
            for l in range(m*p):
                V_test[l][r] = 0
            for k in range(p*n):
                W_test[k][r] = 0
            
            if _verify_decomp(U_test, V_test, W_test, n, m, p, cur_rank):
                # Term r is redundant - actually remove it
                U_cur = []
                for row in U_cur:
                    U_cur.append([row[j] for j in range(cur_rank) if j != r])
                V_cur = []
                for row in V_cur:
                    V_cur.append([row[j] for j in range(cur_rank) if j != r])
                W_cur = []
                for row in W_cur:
                    W_cur.append([row[j] for j in range(cur_rank) if j != r])
                cur_rank -= 1
                eliminated.add(r)
                found = True
                break
        
        if not found:
            break
    
    return U_cur, V_cur, W_cur, cur_rank

def _try_pairwise_combination(U, V, W, n, m, p, rank):
    """
    Try to find if two terms can be combined into one via linear transformation.
    This is a more advanced elimination technique.
    """
    # For now, just try simple elimination
    return _try_eliminate_terms(U, V, W, n, m, p, rank)

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < tol:
                new_row.append(float(round(val)))
            elif abs(val - (round(val*2)/2.0)) < tol:
                new_row.append(round(val*2)/2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 125  # 25s safety margin before 150s limit
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = 60
    best_result = None
    
    # Generate all possible j-pairings and k-pairings
    # m=5: possible pairs
    all_j_pairs_options = []
    # Two pairs from 5 elements
    for j0, j1 in combinations(range(m), 2):
        for j2, j3 in combinations(range(m), 2):
            if j0 < j1 and j2 < j3 and (j0, j1) != (j2, j3):
                # Check no overlap
                if len({j0, j1, j2, j3}) == 4:
                    all_j_pairs_options.append([(j0, j1), (j2, j3)])
    
    # p=6: possible pairs
    all_k_pairs_options = []
    for k0, k1 in combinations(range(p), 2):
        for k2, k3 in combinations(range(p), 2):
            for k4, k5 in combinations(range(p), 2):
                if k0 < k1 and k2 < k3 and k4 < k5:
                    if len({k0, k1, k2, k3, k4, k5}) == 6:
                        # Sort for uniqueness
                        pairs = sorted([(k0, k1), (k2, k3), (k4, k5)])
                        all_k_pairs_options.append(pairs)
    
    # Deduplicate
    all_j_pairs_options = list(set([tuple(tuple(x) for x in opt) for opt in all_j_pairs_options]))
    all_k_pairs_options = list(set([tuple(tuple(x) for x in opt) for opt in all_k_pairs_options]))
    
    # Convert back to lists
    all_j_pairs_options = [list(map(list, opt)) for opt in all_j_pairs_options]
    all_k_pairs_options = [list(map(list, opt)) for opt in all_k_pairs_options]
    
    # Try configurations in order of potential rank
    # Rank = 7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p
    # With 2 j-pairs, 3 k-pairs: rank = 42 + 12 = 54
    # With 1 j-pair, 3 k-pairs: rank = 21 + 24 = 45 (but less efficient)
    
    configurations = []
    
    # All 2-j-pair, 3-k-pair combinations (rank 54)
    for jp in all_j_pairs_options:
        for kp in all_k_pairs_options:
            configurations.append((jp, kp))
    
    # Also try 1 j-pair configurations (rank 45)
    for j0, j1 in combinations(range(m), 2):
        for kp in all_k_pairs_options:
            configurations.append(([ (j0, j1) ], kp))
    
    # Sort by expected rank (lower first)
    def _expected_rank(jp, kp):
        return 7 * len(jp) * len(kp) + 2 * (m - 2*len(jp)) * p
    
    configurations.sort(key=lambda x: _expected_rank(x[0], x[1]))
    
    # Try configurations
    for j_pairs, k_pairs in configurations:
        if _time_left() < 10:
            break
        
        U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs)
        rank = len(U[0])
        
        if not _verify_decomp(U, V, W, n, m, p, rank):
            continue
        
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
        
        # Try to eliminate redundant terms
        if _time_left() > 15:
            U_red, V_red, W_red, red_rank = _try_eliminate_terms(
                U, V, W, n, m, p, rank, max_attempts=100
            )
            
            if red_rank < best_rank:
                best_rank = red_rank
                best_result = {"U": U_red, "V": V_red, "W": W_red}
        
        # Early exit if we found a very good result
        if best_rank <= 50:
            break
    
    # Snap to half-integers for cleaner output
    if best_result is not None:
        best_result["U"] = _snap_to_half_integer(best_result["U"])
        best_result["V"] = _snap_to_half_integer(best_result["V"])
        best_result["W"] = _snap_to_half_integer(best_result["W"])
        
        # Final verification
        if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
            # Fallback to standard
            U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
            best_result = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
