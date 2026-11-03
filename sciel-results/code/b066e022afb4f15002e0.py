"""QAP solver for tho150 using vectorized local search with Hungarian init and SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Convert to float64 for fast arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])
    
    def compute_swap_delta(i, j, p):
        """
        Compute cost delta when swapping facilities at indices i and j in permutation p.
        Uses vectorized operations for speed.
        """
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0.0
        
        # Diagonal terms
        delta = a_f[i, i] * (b_f[lj, lj] - b_f[li, li])
        delta += a_f[j, j] * (b_f[li, li] - b_f[lj, lj])
        
        # Cross terms between i and j
        delta += a_f[i, j] * (b_f[lj, li] - b_f[li, lj])
        delta += a_f[j, i] * (b_f[li, lj] - b_f[lj, li])
        
        # Terms with other facilities k
        # Get all indices except i and j
        if i < j:
            ks = np.concatenate([np.arange(i), np.arange(i+1, j), np.arange(j+1, n)])
        else:
            ks = np.concatenate([np.arange(j), np.arange(j+1, i), np.arange(i+1, n)])
        
        if len(ks) > 0:
            pk = p[ks]
            
            # Vectorized sums
            # a[i, ks] * (b[lj, pk] - b[li, pk])
            delta += np.dot(a_f[i, ks], b_f[lj, pk] - b_f[li, pk])
            # a[ks, i] * (b[pk, lj] - b[pk, li])
            delta += np.dot(a_f[ks, i], b_f[pk, lj] - b_f[pk, li])
            # a[j, ks] * (b[li, pk] - b[lj, pk])
            delta += np.dot(a_f[j, ks], b_f[li, pk] - b_f[lj, pk])
            # a[ks, j] * (b[pk, li] - b[pk, lj])
            delta += np.dot(a_f[ks, j], b_f[pk, li] - b_f[pk, lj])
            
        return delta
    
    def local_search_first_improvement(p, current_cost, end_time, seed=12345):
        """
        Perform First Improvement 2-swap Local Search.
        Randomizes the order of swaps to avoid bias.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        while time.time() < end_time:
            improved = False
            # Create list of all pairs (i, j) with i < j
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                delta = compute_swap_delta(i, j, curr_p)
                
                if delta < 0:
                    # Accept move
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                break
                
        return best_p, best_cost
    
    def simulated_annealing(p, current_cost, end_time, seed=42):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta < 1e-6:
            avg_delta = 1.0
            
        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost
    
    # 1. Hungarian Initialization
    # Minimize sum a[i,j] * b[p[i], p[j]]
    # This is not a simple assignment problem, but we can use a heuristic initialization
    # based on rank matching or spectral methods.
    
    # Rank matching: Heavy facilities (high flow sum) to central locations (low distance sum)
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Also try Hungarian on a relaxed formulation
    # C[i, j] = a[i, :] @ b[:, j] is not quite right for QAP
    # But we can try: minimize sum_i a[i, :] @ b[p[i], :]
    # This is a linear assignment if we approximate b[p[i], :] as independent
    
    # Let's use the rank matching as primary init, and also try a random init
    
    best_p = p_rank.copy()
    best_cost = compute_cost(best_p)
    
    # 2. Polish with Local Search
    end_time_lk1 = deadline - 30.0
    p_lk1, cost_lk1 = local_search_first_improvement(best_p, best_cost, end_time_lk1, seed=12345)
    if cost_lk1 < best_cost:
        best_p = p_lk1
        best_cost = cost_lk1
    
    # 3. Try Random Initialization + Local Search
    if time.time() < deadline - 20.0:
        rng_init = np.random.RandomState(123)
        p_rand = rng_init.permutation(n)
        cost_rand = compute_cost(p_rand)
        
        end_time_lk2 = deadline - 15.0
        p_lk2, cost_lk2 = local_search_first_improvement(p_rand, cost_rand, end_time_lk2, seed=456)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2
            
    # 4. Perturb Best and Local Search
    if time.time() < deadline - 10.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        # Swap 5 random pairs to perturb
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        end_time_lk3 = deadline - 5.0
        p_lk3, cost_lk3 = local_search_first_improvement(p_pert, cost_pert, end_time_lk3, seed=999)
        if cost_lk3 < best_cost:
            best_p = p_lk3
            best_cost = cost_lk3
    
    # 5. Final SA Polish if time remains
    if time.time() < deadline - 2.0:
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 0.5, seed=111)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
