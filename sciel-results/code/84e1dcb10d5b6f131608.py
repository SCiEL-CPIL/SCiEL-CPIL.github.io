"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy Construction (Vectorized Incremental Cost)
    # Sort facilities by total flow descending
    flow_sums = a.sum(axis=1) + a.sum(axis=0)
    order = np.argsort(-flow_sums, kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        f = order[idx]
        if idx == 0:
            # First facility, cost is 0 for all locations
            # Pick location with min sum of distances? Or just first?
            # Let's pick location with min sum of b row
            best_loc = np.argmin(b.sum(axis=1))
        else:
            placed_indices = order[:idx]
            placed_locs = p[placed_indices]
            
            # V = a[f, placed_indices]
            # W = a[placed_indices, f]
            V = a[f, placed_indices]
            W = a[placed_indices, f]
            
            # B_rows = b[:, placed_locs] (N x M)
            # B_cols = b[placed_locs, :] (M x N)
            B_rows = b[:, placed_locs]
            B_cols = b[placed_locs, :]
            
            # C1 = B_rows @ V (N) -> sum_k b[loc, p_k] * a[f, g_k]
            # C2 = W @ B_cols (N) -> sum_k a[g_k, f] * b[p_k, loc]
            C1 = B_rows @ V
            C2 = W @ B_cols
            total_inc = C1 + C2
            
            # Mask used
            total_inc[used] = np.iinfo(np.int64).max
            best_loc = np.argmin(total_inc)
        
        p[f] = best_loc
        used[best_loc] = True

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = total_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    t0 = time.time()
    time_limit = 110.0

    # --- Iterated Local Search with Tabu Search ---
    rng = np.random.default_rng(42)
    
    def calc_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        diff_row = a[i, :] - a[j, :]
        diff_col = a[:, i] - a[:, j]
        
        b_v_row = b[v, p]
        b_u_row = b[u, p]
        b_col_v = b[p, v]
        b_col_u = b[p, u]
        
        s1 = np.dot(diff_row, b_v_row - b_u_row)
        s2 = np.dot(diff_col, b_col_v - b_col_u)
        
        s1 -= diff_row[i] * (b[v, u] - b[u, u])
        s2 -= diff_col[i] * (b[u, v] - b[u, u])
        s1 -= diff_row[j] * (b[v, v] - b[u, v])
        s2 -= diff_col[j] * (b[v, v] - b[v, u])
        
        direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        return s1 + s2 + direct

    tabu_list = deque()
    tabu_len = 30
    max_no_improve = 100
    no_improve_count = 0
    sample_size = 50
    
    while time.time() - t0 < time_limit:
        # Local Search: Tabu
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            best_move = None
            best_delta = np.iinfo(np.int64).max
            best_i, best_j = -1, -1
            
            # Sample random moves
            pairs = set()
            for _ in range(sample_size):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    if i > j: i, j = j, i
                    pairs.add((i, j))
            
            for i, j in pairs:
                # Check tabu
                is_tabu = False
                for t_i, t_j in tabu_list:
                    if (t_i == i and t_j == j) or (t_i == j and t_j == i):
                        is_tabu = True
                        break
                
                delta = calc_delta(i, j, p, a, b)
                
                # Aspiration: accept if it improves global best
                if delta < best_cost - current_cost:
                    is_tabu = False
                
                if not is_tabu:
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                        best_i, best_j = i, j
            
            if best_move is not None:
                i, j = best_move
                p[i], p[j] = p[j], p[i]
                current_cost += best_delta
                
                tabu_list.append((i, j))
                if len(tabu_list) > tabu_len:
                    tabu_list.popleft()
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                improved = True
                if no_improve_count > max_no_improve:
                    break
            else:
                break
        
        # Perturbation
        if time.time() - t0 >= time_limit:
            break
            
        num_perturb = 5
        for _ in range(num_perturb):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = total_cost(p)
        no_improve_count = 0
        tabu_list = deque()

    return best_p.tolist()
# EVOLVE-BLOCK-END
