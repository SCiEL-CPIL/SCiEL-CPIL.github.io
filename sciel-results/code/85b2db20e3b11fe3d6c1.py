import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the cost change for all pairwise swaps in O(n^2) using vectorization.
    Returns a matrix D where D[i,j] is the cost change for swapping p[i] and p[j].
    """
    n = len(p)
    # We want to compute delta for swapping i and j.
    # Let u = p[i], v = p[j].
    # The change in cost is:
    # sum_k a[i,k](b[v,p[k]] - b[u,p[k]]) + sum_k a[j,k](b[u,p[k]] - b[v,p[k]])
    # + terms involving i and j directly.
    
    # Let's define:
    # V_j = b[p[j], p] - b[p[i], p]  (vector of size n)
    # W_j = a[i, :] - a[j, :]        (vector of size n)
    # The main term is sum_k W_j[k] * V_j[k].
    # But we must exclude k=i and k=j from this sum because the direct terms are handled separately.
    # Actually, let's derive the full formula carefully.
    
    # Cost = sum_{x,y} a[x,y] b[p[x], p[y]]
    # Swap i and j.
    # Change = sum_y a[i,y](b[v, p[y]] - b[u, p[y]]) + sum_y a[j,y](b[u, p[y]] - b[v, p[y]])
    #        + (terms where x=i, y=i or j) + (terms where x=j, y=i or j)
    
    # Let's group by pairs (x,y).
    # Pairs not involving i or j: no change.
    # Pairs (i, k) where k not i,j: a[i,k](b[v, p[k]] - b[u, p[k]])
    # Pairs (j, k) where k not i,j: a[j,k](b[u, p[k]] - b[v, p[k]])
    # Pairs (k, i) where k not i,j: a[k,i](b[p[k], v] - b[p[k], u])
    # Pairs (k, j) where k not i,j: a[k,j](b[p[k], u] - b[p[k], v])
    # Pairs (i, i): a[i,i](b[v, v] - b[u, u])
    # Pairs (j, j): a[j,j](b[u, u] - b[v, v])
    # Pairs (i, j): a[i,j](b[v, u] - b[u, v])
    # Pairs (j, i): a[j,i](b[u, v] - b[v, u])
    
    # Let's combine the sums over k not i,j.
    # Sum1 = sum_{k not i,j} [ a[i,k] - a[j,k] ] * (b[v, p[k]] - b[u, p[k]])
    # Sum2 = sum_{k not i,j} [ a[k,i] - a[k,j] ] * (b[p[k], v] - b[p[k], u])
    
    # Note that b[v, p[k]] - b[u, p[k]] is the k-th element of (b[v, p] - b[u, p]).
    # And b[p[k], v] - b[p[k], u] is the k-th element of (b[p, v] - b[p, u]).
    
    # Let V = b[v, p] - b[u, p]
    # Let W = b[p, v] - b[p, u]
    # Let A_diff_row = a[i, :] - a[j, :]
    # Let A_diff_col = a[:, i] - a[:, j]
    
    # Sum1 = dot(A_diff_row, V) - A_diff_row[i]*V[i] - A_diff_row[j]*V[j]
    # Sum2 = dot(A_diff_col, W) - A_diff_col[i]*W[i] - A_diff_col[j]*W[j]
    
    # Direct terms:
    # D_ii = a[i,i](b[v,v] - b[u,u])
    # D_jj = a[j,j](b[u,u] - b[v,v])
    # D_ij = a[i,j](b[v,u] - b[u,v])
    # D_ji = a[j,i](b[u,v] - b[v,u])
    
    # Total Delta = Sum1 + Sum2 + D_ii + D_jj + D_ij + D_ji
    
    # This is still O(n) per pair if done naively.
    # But we can vectorize over j for a fixed i?
    # For fixed i, we want Delta(i, j) for all j.
    # u = p[i] is fixed. v = p[j] varies.
    # V_j = b[p[j], p] - b[u, p]
    # W_j = b[p, p[j]] - b[p, u]
    # A_diff_row_j = a[i, :] - a[j, :]
    # A_diff_col_j = a[:, i] - a[:, j]
    
    # This seems hard to fully vectorize in 2D without large memory usage.
    # Let's stick to the O(n) delta calculation but optimize it with NumPy.
    # The current _swap_delta is already using np.dot which is fast.
    # The bottleneck in the previous program was the Python loop overhead.
    # We can try to reduce the number of calls or use a smarter search.
    
    # Alternative: Tabu Search.
    # Tabu search can escape local minima better than 2-opt.
    # We will implement a simple Tabu Search with a vectorized delta calculation if possible, 
    # or just efficient O(n) delta calculation.
    
    pass

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    O(n) time complexity.
    """
    pi = p[i]
    pj = p[j]
    
    # Precompute vectors
    b_pi_p = b[pi, p]
    b_pj_p = b[pj, p]
    b_p_pi = b[p, pi]
    b_p_pj = b[p, pj]
    
    # Sums over k not i, j
    # Sum1 = sum_{k not i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_row = a[i, :] - a[j, :]
    diff_b_row = b_pj_p - b_pi_p
    sum1_total = np.dot(diff_row, diff_b_row)
    sum1 = sum1_total - diff_row[i]*diff_b_row[i] - diff_row[j]*diff_b_row[j]
    
    # Sum2 = sum_{k not i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_col = a[:, i] - a[:, j]
    diff_b_col = b_p_pj - b_p_pi
    sum2_total = np.dot(diff_col, diff_b_col)
    sum2 = sum2_total - diff_col[i]*diff_b_col[i] - diff_col[j]*diff_b_col[j]
    
    # Direct terms
    b_pi_pi = b[pi, pi]
    b_pj_pj = b[pj, pj]
    b_pi_pj = b[pi, pj]
    b_pj_pi = b[pj, pi]
    
    d_ii = a[i,i] * (b_pj_pj - b_pi_pi)
    d_jj = a[j,j] * (b_pi_pi - b_pj_pj)
    d_ij = a[i,j] * (b_pj_pi - b_pi_pj)
    d_ji = a[j,i] * (b_pi_pj - b_pj_pi)
    
    return int(sum1 + sum2 + d_ii + d_jj + d_ij + d_ji)

def _tabu_search(a, b, p, current_cost, time_limit=10.0, tabu_size=50):
    """
    Simple Tabu Search for QAP.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Tabu list: store (i, j) pairs that are tabu.
    # We use a dictionary to store the iteration number when they become non-tabu.
    tabu_dict = {}
    current_iter = 0
    
    while time.time() - start_time < time_limit:
        current_iter += 1
        
        # Find best non-tabu move
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # To speed up, we can sample a subset of moves if n is large, 
        # but for n=100, O(n^2) = 10000 evaluations per iteration is feasible.
        # Each evaluation is O(n)=100 ops. So 1e6 ops per iteration.
        # 1e6 ops is ~1ms in NumPy. So 1000 iterations per second.
        # 10 seconds -> 10000 iterations.
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                # Check tabu
                key = (i, j)
                if key in tabu_dict and tabu_dict[key] > current_iter:
                    continue
                
                delta = _swap_delta(a, b, p, i, j)
                
                # Aspiration criterion: accept tabu move if it improves best cost
                if delta < best_delta or (key in tabu_dict and tabu_dict[key] <= current_iter and cost + delta < best_cost):
                    # Wait, aspiration is usually: if cost + delta < best_cost, accept even if tabu.
                    # But here we are selecting the best move.
                    # Let's just find the best non-tabu move first.
                    if key not in tabu_dict or tabu_dict[key] <= current_iter:
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
        
        # If no non-tabu move found, or best_delta is not negative (but we allow worsening moves in TS)
        # In standard TS, we accept the best move even if it worsens cost, as long as it's not tabu.
        # If all moves are tabu, we might need to clear tabu list or use aspiration.
        # Let's assume we always find a non-tabu move.
        
        if best_i != -1:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            
            # Update tabu list
            tabu_dict[(best_i, best_j)] = current_iter + tabu_size
            # Also, the reverse swap is the same, so (best_j, best_i) is effectively tabu too.
            # But we only consider i < j, so we just store (min, max).
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
        else:
            # All moves tabu? Relax tabu list or break.
            break
            
    return best_p, best_cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def _or_opt_delta(a, b, p, i, j):
    """
    Compute cost change for moving facility i to position j (shifting others).
    This is more complex. We will stick to 2-opt and Tabu for now.
    """
    pass

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin from 150s
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    # Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Random
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = min(cost_greedy, cost_random)
    
    # Phase 2: Initial Polish with Tabu Search
    elapsed = time.time() - start_time
    ts_time = min(10.0, total_deadline - elapsed - 20.0)
    if ts_time > 0:
        best_p, best_cost = _tabu_search(a, b, best_p, best_cost, time_limit=ts_time, tabu_size=50)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 5.0 # Reserve 5s for final TS
    num_sa_runs = 5
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        # Diversify start
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            p_start = np.random.permutation(n).astype(np.int64)
            c_start = _qap_cost(a, b, p_start)
            # Quick polish with Tabu
            p_start, c_start = _tabu_search(a, b, p_start, c_start, time_limit=2.0, tabu_size=30)
        
        T_start = 1000.0
        T_end = 1.0
        p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
            
    # Phase 4: Final Polish with Tabu Search
    elapsed = time.time() - start_time
    final_ts_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ts_time > 0:
        best_p, best_cost = _tabu_search(a, b, best_p, best_cost, time_limit=final_ts_time, tabu_size=50)
        
    return best_p.tolist()

