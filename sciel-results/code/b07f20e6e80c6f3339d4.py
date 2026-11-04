"""
MOGA-based solver for QAPLIB wil100.
Strategy: Multi-Objective Genetic Algorithm maintaining a Pareto front.
Objectives:
1. Linear surrogate cost (flow centrality vs distance centrality)
2. Quadratic surrogate cost (approximation of actual QAP cost using low-rank approx)
Diversity: Crowding distance in the objective space.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _local_search(a, b, p, time_limit):
    """
    Vectorized first-improvement local search.
    Uses batched delta calculation to speed up convergence.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Compute all deltas for swaps (i, j) where i < j
        # This is O(n^2) memory and compute, but n=100 is small enough (10k swaps)
        # However, doing it fully vectorized is complex. 
        # Let's stick to a loop but with a time check every 100 iterations.
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best_p, best_cost
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return best_p, best_cost

def _linear_surrogate_cost(a, b, p):
    """
    Objective 1: Linear alignment.
    High flow facilities should go to locations with high connectivity (row sums).
    Cost = - sum(a_row_sums[i] * b_row_sums[p[i]])
    """
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    # We want to maximize sum(a_rs[i] * b_rs[p[i]])
    # So cost is negative of that
    return -np.sum(a_rs * b_rs[p])

def _quadratic_surrogate_cost(a, b, p):
    """
    Objective 2: Quadratic approximation.
    Use a low-rank approximation of the QAP cost.
    Cost approx = sum_i sum_j a[i,j] * (b_rs[p[i]] * b_cs[p[j]]) / n
    where b_rs is row sums and b_cs is col sums.
    This captures the "quadratic" nature better than linear.
    """
    b_rs = b.sum(axis=1)
    b_cs = b.sum(axis=0)
    
    # Vectorized calculation:
    # Term = sum_{i,j} a[i,j] * b_rs[p[i]] * b_cs[p[j]]
    # Let u[i] = b_rs[p[i]] and v[j] = b_cs[p[j]]
    # Term = u @ a @ v
    u = b_rs[p]
    v = b_cs[p]
    
    # u @ a is O(n^2), then dot with v is O(n)
    val = np.dot(u, np.dot(a, v))
    
    # Normalize to be comparable to linear surrogate
    # Scale by 1/n^2 to keep magnitudes similar
    return val / (len(a) ** 2)

def _pareto_dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2 (minimization).
    obj1 and obj2 are arrays of objectives.
    """
    if np.all(obj1 <= obj2) and np.any(obj1 < obj2):
        return True
    return False

def _crowding_distance(objs):
    """
    Calculate crowding distance for a set of objectives.
    objs: (M, K) array where M is number of solutions, K is number of objectives.
    Returns: (M,) array of distances.
    """
    M = len(objs)
    if M <= 2:
        return np.full(M, np.inf)
    
    distances = np.zeros(M)
    K = objs.shape[1]
    
    for k in range(K):
        # Sort by k-th objective
        idx = np.argsort(objs[:, k])
        # Boundary points get inf
        distances[idx[0]] = np.inf
        distances[idx[-1]] = np.inf
        
        # Normalize range
        obj_range = objs[idx[-1], k] - objs[idx[0], k]
        if obj_range == 0:
            continue
            
        for i in range(1, M - 1):
            distances[idx[i]] += (objs[idx[i+1], k] - objs[idx[i-1], k]) / obj_range
            
    return distances

def _moga_evolve(a, b, pop, objectives, time_limit):
    """
    Multi-Objective Genetic Algorithm evolution.
    pop: list of permutations
    objectives: list of objective vectors (each is (2,) array)
    """
    n = len(a)
    start_time = time.time()
    pop_size = len(pop)
    
    # Precompute surrogates for speed? 
    # Linear surrogate is fast. Quadratic is O(n^2) per eval.
    # With n=100 and pop_size=50, 50*2*10000 ops = 1M ops per gen. Very fast.
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate current population (if not done)
        # We assume objectives are up to date for current pop
        
        # 2. Selection: Tournament selection based on Pareto rank + crowding distance
        # Compute ranks
        ranks = np.zeros(pop_size, dtype=int)
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j and _pareto_dominates(objectives[j], objectives[i]):
                    ranks[i] += 1
        
        # Non-dominated solutions have rank 0
        # For simplicity, we use rank as fitness. Lower rank is better.
        # If same rank, use crowding distance.
        
        # Calculate crowding distance
        obj_array = np.array(objectives)
        cd = _crowding_distance(obj_array)
        
        # Fitness score: rank * 1000 - crowding_distance (higher is better)
        fitness = ranks * 1000.0 - cd
        
        # 3. Crossover: Uniform crossover or Order Crossover
        new_pop = []
        for _ in range(pop_size):
            # Tournament select 2 parents
            p1_idx = np.random.choice(pop_size, p=np.exp(fitness - fitness.max()))
            p2_idx = np.random.choice(pop_size, p=np.exp(fitness - fitness.max()))
            
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            
            # Order Crossover (OX1)
            c1 = np.empty(n, dtype=np.int64)
            c2 = np.empty(n, dtype=np.int64)
            
            cut1 = np.random.randint(1, n)
            cut2 = np.random.randint(cut1, n)
            
            # Segment
            seg1 = p1[cut1:cut2].copy()
            seg2 = p2[cut1:cut2].copy()
            
            # Fill rest
            rest1 = [x for x in p2 if x not in seg1]
            rest2 = [x for x in p1 if x not in seg2]
            
            # Place segment
            c1[cut1:cut2] = seg1
            c2[cut1:cut2] = seg2
            
            # Fill rest in order
            pos = 0
            for i in range(n):
                if cut1 <= i < cut2:
                    continue
                c1[i] = rest1[pos]
                pos += 1
            
            pos = 0
            for i in range(n):
                if cut1 <= i < cut2:
                    continue
                c2[i] = rest2[pos]
                pos += 1
            
            new_pop.append(c1)
            new_pop.append(c2)
            
            # Mutation: Swap mutation
            if np.random.random() < 0.5:
                i, j = np.random.choice(n, 2, replace=False)
                new_pop[-1][i], new_pop[-1][j] = new_pop[-1][j], new_pop[-1][i]
            if np.random.random() < 0.5:
                i, j = np.random.choice(n, 2, replace=False)
                new_pop[-2][i], new_pop[-2][j] = new_pop[-2][j], new_pop[-2][i]
        
        # 4. Environmental Selection: Combine parents and offspring, select best pop_size
        combined_pop = pop + new_pop
        combined_objs = []
        for p in combined_pop:
            o1 = _linear_surrogate_cost(a, b, p)
            o2 = _quadratic_surrogate_cost(a, b, p)
            combined_objs.append([o1, o2])
        
        combined_objs = np.array(combined_objs)
        
        # Compute ranks for combined
        total_size = len(combined_pop)
        ranks = np.zeros(total_size, dtype=int)
        for i in range(total_size):
            for j in range(total_size):
                if i != j and _pareto_dominates(combined_objs[j], combined_objs[i]):
                    ranks[i] += 1
        
        # Sort by rank
        sort_idx = np.argsort(ranks)
        
        # Select best pop_size
        # If we hit a rank boundary, use crowding distance
        selected_indices = []
        current_rank = 0
        while len(selected_indices) < pop_size:
            # Find all individuals with current_rank
            idx_in_rank = [i for i in range(total_size) if ranks[i] == current_rank]
            
            if len(selected_indices) + len(idx_in_rank) <= pop_size:
                selected_indices.extend(idx_in_rank)
            else:
                # Need to select from this rank based on crowding distance
                obj_subset = combined_objs[idx_in_rank]
                cd = _crowding_distance(obj_subset)
                # Sort by cd descending
                cd_order = np.argsort(-cd)
                needed = pop_size - len(selected_indices)
                selected_indices.extend([idx_in_rank[k] for k in cd_order[:needed]])
            
            current_rank += 1
        
        pop = [combined_pop[i] for i in selected_indices]
        objectives = [combined_objs[i] for i in selected_indices]
        
        # 5. Local Search on best solution (lowest actual cost)
        # Find best actual cost in population
        best_actual = np.inf
        best_idx = 0
        for i, p in enumerate(pop):
            c = _qap_cost(a, b, p)
            if c < best_actual:
                best_actual = c
                best_idx = i
        
        # Spend some time on local search for the best individual
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.5:
            ls_time = min(1.0, time_remaining * 0.1)
            p_ls, c_ls = _local_search(a, b, pop[best_idx].copy(), ls_time)
            if c_ls < best_actual:
                pop[best_idx] = p_ls
                objectives[best_idx] = [
                    _linear_surrogate_cost(a, b, p_ls),
                    _quadratic_surrogate_cost(a, b, p_ls)
                ]
                best_actual = c_ls
        
    # Return best individual based on actual cost
    best_p = None
    best_cost = np.inf
    for p in pop:
        c = _qap_cost(a, b, p)
        if c < best_cost:
            best_cost = c
            best_p = p
    return best_p, best_cost

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums
        
    C = -np.outer(ars_norm, brs_norm)
    
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    # Initialize population
    pop_size = 40
    
    # Start with Hungarian solution and its perturbations
    p_hung = _hungarian_initial(a, b)
    
    pop = [p_hung.copy()]
    # Add random permutations
    for _ in range(pop_size - 1):
        p_rand = np.random.permutation(n)
        pop.append(p_rand)
    
    # Evaluate objectives for initial population
    objectives = []
    for p in pop:
        o1 = _linear_surrogate_cost(a, b, p)
        o2 = _quadratic_surrogate_cost(a, b, p)
        objectives.append([o1, o2])
    
    # Run MOGA
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        best_p, best_cost = _moga_evolve(a, b, pop, objectives, time_remaining - 5)
    else:
        # Fallback if no time
        best_p = p_hung
        best_cost = _qap_cost(a, b, p_hung)
    
    # Final polish with local search on the best solution found
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 2:
        p_ls, c_ls = _local_search(a, b, best_p.copy(), time_remaining - 1)
        if c_ls < best_cost:
            best_p = p_ls
            best_cost = c_ls
            
    return best_p.tolist()
