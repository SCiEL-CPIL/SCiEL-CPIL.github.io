import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def make_B_perm(p):
        return b_f[np.ix_(p, p)].copy()

    def swap_in_B_perm(B_perm, i, j):
        B_perm[[i, j], :] = B_perm[[j, i], :]
        B_perm[:, [i, j]] = B_perm[:, [j, i]]

    def local_search(p, B_perm, end_time, max_passes=1000):
        curr_p = p.copy()
        curr_cost = compute_cost(curr_p)
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < end_time:
            improved = False
            passes += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > end_time:
                        return curr_p, curr_cost, B_perm
                    
                    drow = B_perm[j, :] - B_perm[i, :]
                    dcol = B_perm[:, j] - B_perm[:, i]
                    a_diff_row = a_f[i, :] - a_f[j, :]
                    a_diff_col = a_f[:, i] - a_f[:, j]
                    
                    delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                    
                    if delta < -1e-10:
                        curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                        curr_cost += delta
                        swap_in_B_perm(B_perm, i, j)
                        improved = True
                        
        return curr_p, curr_cost, B_perm

    def spectral_init(A, B):
        try:
            wA, vA = np.linalg.eigh(A)
            wB, vB = np.linalg.eigh(B)
            
            k = min(10, n)
            idxA = np.argsort(wA)[::-1][:k]
            idxB = np.argsort(wB)[::-1][:k]
            
            X = vA[:, idxA]
            Y = vB[:, idxB]
            
            U, S, Vt = np.linalg.svd(Y.T @ X)
            Q = U @ Vt
            X_aligned = X @ Q
            
            X_norms = np.sum(X_aligned ** 2, axis=1)
            Y_norms = np.sum(Y ** 2, axis=1)
            Cross = X_aligned @ Y.T
            C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
            
            row_ind, col_ind = linear_sum_assignment(C)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
        except Exception:
            return None

    def greedy_init(A, B):
        p = np.full(n, -1, dtype=np.int64)
        unassigned_fac = set(range(n))
        unassigned_loc = set(range(n))
        
        flow_sums = A.sum(axis=1)
        order = np.argsort(-flow_sums)
        
        assigned_fac = []
        assigned_loc = []
        
        for i in order:
            if i not in unassigned_fac:
                continue
            best_loc = -1
            
            if not assigned_fac:
                b_row_sums = B.sum(axis=1)
                best_loc = np.argmin(b_row_sums)
            else:
                unassigned_loc_list = list(unassigned_loc)
                B_unassigned = B[unassigned_loc_list, :]
                
                A_i_ass = A[i, assigned_fac]
                A_ass_i = A[assigned_fac, i]
                P_ass = np.array(assigned_loc)
                
                B_ass = B[P_ass, :]
                
                term1 = B_unassigned[:, P_ass] @ A_i_ass
                term2 = B_ass.T @ A_ass_i
                
                total_costs = term1 + term2
                best_idx_in_unassigned = np.argmin(total_costs)
                best_loc = unassigned_loc_list[best_idx_in_unassigned]
            
            p[i] = best_loc
            unassigned_fac.remove(i)
            unassigned_loc.remove(best_loc)
            assigned_fac.append(i)
            assigned_loc.append(best_loc)
            
        return p

    # --- Phase 1: Initialization ---
    candidates = []
    
    if time.time() < deadline - 10.0:
        p_spec = spectral_init(a_f, b_f)
        if p_spec is not None:
            candidates.append(p_spec)
            
    if time.time() < deadline - 10.0:
        p_greedy = greedy_init(a_f, b_f)
        candidates.append(p_greedy)
        
    rng = np.random.RandomState(42)
    p_rand = rng.permutation(n)
    candidates.append(p_rand)
    
    best_init_p = None
    best_init_cost = np.inf
    best_init_B_perm = None
    
    for p_cand in candidates:
        if time.time() > deadline - 10.0:
            break
        B_perm_cand = make_B_perm(p_cand)
        p_quick, cost_quick, B_perm_quick = local_search(p_cand, B_perm_cand, time.time() + 2.0, max_passes=1)
        if cost_quick < best_init_cost:
            best_init_cost = cost_quick
            best_init_p = p_quick
            best_init_B_perm = B_perm_quick
            
    if best_init_p is None:
        best_init_p = p_rand
        best_init_cost = compute_cost(best_init_p)
        best_init_B_perm = make_B_perm(best_init_p)

    # --- Phase 2: Iterated Local Search ---
    current_p = best_init_p.copy()
    current_cost = best_init_cost
    current_B_perm = best_init_B_perm.copy()
    
    best_p = best_init_p.copy()
    best_cost = best_init_cost
    
    perturbation_strength = 5
    
    while time.time() < deadline - 2.0:
        current_p, current_cost, current_B_perm = local_search(current_p, current_B_perm, deadline - 2.0)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            
        for _ in range(perturbation_strength):
            i, j = rng.choice(n, 2, replace=False)
            current_p[i], current_p[j] = current_p[j], current_p[i]
            swap_in_B_perm(current_B_perm, i, j)
            
    return best_p.tolist()
