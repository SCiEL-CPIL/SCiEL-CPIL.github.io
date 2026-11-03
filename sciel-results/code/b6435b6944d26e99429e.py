"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 150s limit (extra buffer)
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    def get_delta_fast(i, j, p, B_p):
        u = p[i]
        v = p[j]
        
        da_row = a[i] - a[j]
        da_col = a[:, i] - a[:, j]
        db_row = B_p[v] - B_p[u]
        db_col = B_p[:, v] - B_p[:, u]
        
        main_term = np.dot(da_row, db_row) + np.dot(da_col, db_col)
        
        b_uu = b[u, u]
        b_vv = b[v, v]
        b_uv = b[u, v]
        b_vu = b[v, u]
        
        ki_contrib = (a[i, i] - a[j, i]) * (b_vu - b_uu) + (a[i, i] - a[j, i]) * (b_uv - b_uu)
        kj_contrib = (a[i, j] - a[j, j]) * (b_vv - b_uv) + (a[i, j] - a[j, j]) * (b_vv - b_vu)
        
        direct = (a[i, i] * (b_vv - b_uu) + a[j, j] * (b_uu - b_vv) + 
                  a[i, j] * (b_vu - b_uv) + a[j, i] * (b_uv - b_vu))
        
        delta = main_term - ki_contrib - kj_contrib + direct
        return int(delta)
    
    # Initialization Strategies
    
    def init_rank():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_rank_reversed():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")[::-1]
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_spectral():
        try:
            eigvals, eigvecs = np.linalg.eigh(b.astype(np.float64))
            idx = np.argmin(np.abs(eigvals))
            loc_scores = eigvecs[:, idx]
            loc_order = np.argsort(loc_scores, kind="stable")
            
            a_row_sums = a.sum(axis=1)
            fac_order = np.argsort(a_row_sums, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except:
            return np.random.permutation(n).astype(np.int64)

    def init_random():
        return np.random.permutation(n).astype(np.int64)

    def init_hungarian():
        """
        Hungarian algorithm initialization.
        Formulate a bipartite matching where edge weight (i, j) approximates
        the contribution of placing facility i at location j.
        We use the sum of interactions with all other facilities/locations.
        weight[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        This captures the linearized pairwise affinity.
        """
        try:
            # Compute interaction matrix C where C[i, j] is the cost of placing i at j
            # C[i, j] = sum_k (a[i, k] * b[j, k] + a[k, i] * b[k, j])
            # This is (A @ B.T)[i, j] + (A.T @ B)[i, j]
            # Note: A and B are int64, but for Hungarian we can use float64
            # We want to minimize, so we pass the cost matrix directly.
            
            # A @ B.T gives sum_k a[i, k] * b[j, k] (since B.T[j, k] = B[k, j]? No, B.T[j, k] is B[k, j] if we index correctly)
            # Let's be precise:
            # Term 1: sum_k a[i, k] * b[j, k]. This is row i of A dotted with row j of B.
            # Let M1 = A @ B.T. M1[i, j] = sum_k A[i, k] * B.T[k, j] = sum_k A[i, k] * B[j, k]. Correct.
            
            # Term 2: sum_k a[k, i] * b[k, j]. This is col i of A dotted with col j of B.
            # Let M2 = A.T @ B. M2[i, j] = sum_k A.T[i, k] * B[k, j] = sum_k A[k, i] * B[k, j]. Correct.
            
            # Total linearized cost
            cost_mat = (a.astype(np.float64) @ b.astype(np.float64).T + 
                        a.astype(np.float64).T @ b.astype(np.float64))
            
            # Solve Hungarian
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            # row_ind are facilities, col_ind are locations
            # p[facility] = location
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
        except:
            return np.random.permutation(n).astype(np.int64)

    def local_search(p, current_cost, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        best_p = p.copy()
        best_cost = current_cost
        B_p = b[:, p].copy()
        
        tabu_tenure = 15
        tabu_list = {}
        iteration = 0
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < max_time:
            best_delta = np.inf
            best_i = -1
            best_j = -1
            is_aspiration = False
            
            for i in range(n):
                for j in range(i + 1, n):
                    move_key = (i, j)
                    tabu_until = tabu_list.get(move_key, -1)
                    
                    if tabu_until > iteration:
                        delta = get_delta_fast(i, j, p, B_p)
                        new_cost = current_cost + delta
                        if new_cost < best_cost:
                            is_aspiration = True
                        else:
                            continue
                    else:
                        is_aspiration = False
                    
                    if not is_aspiration:
                        delta = get_delta_fast(i, j, p, B_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        is_aspiration = False
            
            if best_i == -1:
                # No improving move found, pick random non-tabu
                valid_moves = []
                for i in range(n):
                    for j in range(i + 1, n):
                        move_key = (i, j)
                        if tabu_list.get(move_key, -1) <= iteration:
                            valid_moves.append((i, j))
                if not valid_moves:
                    valid_moves = [(i, j) for i in range(n) for j in range(i + 1, n)]
                
                best_i, best_j = random.choice(valid_moves)
                best_delta = get_delta_fast(best_i, best_j, p, B_p)
            
            # Perform move
            u_old = p[best_i]
            v_old = p[best_j]
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            # Update B_p
            B_p[:, best_i] = b[:, v_old]
            B_p[:, best_j] = b[:, u_old]
            B_p[u_old, :] = b[u_old, p]
            B_p[v_old, :] = b[v_old, p]
            
            B_p[u_old, best_i] = b[u_old, v_old]
            B_p[u_old, best_j] = b[u_old, u_old]
            B_p[v_old, best_i] = b[v_old, v_old]
            B_p[v_old, best_j] = b[v_old, u_old]
            
            tabu_list[(best_i, best_j)] = iteration + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            iteration += 1
            
            if no_improve > max_no_improve:
                num_perturb = random.randint(3, 6)
                for _ in range(num_perturb):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                B_p = b[:, p].copy()
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
                tabu_list.clear()
        
        return best_p, best_cost

    # Multi-start Local Search
    best_overall_p = None
    best_overall_cost = np.inf
    
    # Prioritize Hungarian init as it captures global structure
    inits = [
        init_hungarian(),
        init_rank(),
        init_rank_reversed(),
        init_spectral(),
        init_random()
    ]
    
    for init_p in inits:
        if time.time() > deadline - 15.0:
            break
        
        init_cost = calc_cost(init_p)
        if init_cost < best_overall_cost:
            best_overall_cost = init_cost
            best_overall_p = init_p.copy()
        
        # Run local search
        ls_p, ls_cost = local_search(init_p, init_cost)
        
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_p = ls_p.copy()
    
    # Final polish
    if best_overall_p is not None and time.time() < deadline - 10.0:
        final_p, final_cost = local_search(best_overall_p.copy(), best_overall_cost, max_time=deadline - 5.0)
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_p = final_p
    
    return best_overall_p.tolist()

# EVOLVE-BLOCK-END
