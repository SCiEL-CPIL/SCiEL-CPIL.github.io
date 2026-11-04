import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank Matching
    # Heavy flows (high row sum in A) to short distances (low row sum in B)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        B_perm = b[np.ix_(p, p)]
        return np.sum(a * B_perm)

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu Search Setup
    start_time = time.time()
    time_limit = 115.0
    
    # Tabu matrix: stores the iteration index until which the move is tabu
    tabu_matrix = np.zeros((n, n), dtype=np.int64)
    tabu_tenure = 100  # Number of iterations to keep a move tabu
    
    rng = np.random.default_rng(42)
    
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        iter_count += 1
        
        # Select a non-tabu move
        # Try a few random pairs
        move_found = False
        i = j = -1
        for _ in range(10):
            ri = rng.integers(n)
            rj = rng.integers(n)
            if ri == rj:
                continue
            if tabu_matrix[ri, rj] <= iter_count:
                i, j = ri, rj
                move_found = True
                break
        
        if not move_found:
            # If stuck, just pick any random move (ignoring tabu) to escape
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue

        u = p[i]
        v = p[j]
        
        # Calculate Delta
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        
        S = np.dot(diff_a, diff_b)
        
        term_i = diff_a[i] * diff_b[i]
        term_j = diff_a[j] * diff_b[j]
        
        delta = S - term_i - term_j
        
        # Aspiration Criterion: Accept if it improves best_cost
        # Or if it's not tabu and improves current_cost
        # Standard Tabu: Accept if (not tabu AND improves) OR (improves best)
        
        is_tabu = tabu_matrix[i, j] > iter_count
        improves_current = delta < 0
        improves_best = current_cost + delta < best_cost
        
        if improves_current or improves_best:
            # Accept move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            # Set tabu for this move
            tabu_matrix[i, j] = iter_count + tabu_tenure
            tabu_matrix[j, i] = iter_count + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Move is not improving current or best.
            # Skip it.
            no_improve_count += 1
            
            # If stuck for too long, randomize to escape local minimum
            if no_improve_count > 5000:
                for _ in range(10):
                    ri = rng.integers(n)
                    rj = rng.integers(n)
                    if ri != rj:
                        ui = p[ri]
                        vj = p[rj]
                        da = a[ri] - a[rj]
                        db = b[vj, p] - b[ui, p]
                        S_rand = np.dot(da, db)
                        ti = da[ri] * db[ri]
                        tj = da[rj] * db[rj]
                        d_rand = S_rand - ti - tj
                        p[ri], p[rj] = p[rj], p[ri]
                        current_cost += d_rand
                no_improve_count = 0

    # Final Polish: Best Improvement Local Search
    # Use best_p as starting point for polish
    p_polish = best_p.copy()
    cost_polish = best_cost
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                u = p_polish[i]
                v = p_polish[j]
                diff_a = a[i] - a[j]
                diff_b = b[v, p_polish] - b[u, p_polish]
                S = np.dot(diff_a, diff_b)
                term_i = diff_a[i] * diff_b[i]
                term_j = diff_a[j] * diff_b[j]
                delta = S - term_i - term_j
                
                if delta < 0:
                    p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
                    cost_polish += delta
                    improved = True
                    if cost_polish < best_cost:
                        best_cost = cost_polish
                        best_p = p_polish.copy()
        
        if time.time() - start_time > time_limit:
            break

    return best_p.tolist()
