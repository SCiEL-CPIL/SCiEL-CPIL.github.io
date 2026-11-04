"""
Multi-Objective Genetic Algorithm (MOGA) for QAPLIB wil100.
Maintains a Pareto front of non-dominated solutions based on surrogate objectives
to explore the search space more broadly than single-objective SA.
"""
# EVOLVE-BLOCK-START
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
    
    # Diagonal terms (usually 0 for QAP but handle generally)
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
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _hungarian_initial(a, b, objective='linear'):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    objective: 'linear' uses row-sum centrality, 'quadratic' uses a quadratic penalty approx.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    if objective == 'linear':
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
    else: # quadratic approximation
        # Approximate quadratic interaction by matching high-flow pairs with low-distance pairs
        # This is a heuristic surrogate.
        # Use a random projection or simplified interaction term.
        # For simplicity, we use a weighted combination of row sums and a random projection.
        # A better surrogate: C[i,j] = sum_k a[i,k] * b[j, k] is not linear in assignment.
        # We stick to a robust linear surrogate but with different weighting for diversity.
        # Let's use column sums for variety or a different normalization.
        a_col_sums = a.sum(axis=0)
        b_col_sums = b.sum(axis=0)
        ars_max = np.max(np.abs(a_col_sums))
        brs_max = np.max(np.abs(b_col_sums))
        if ars_max > 0:
            ars_norm = a_col_sums / ars_max
        else:
            ars_norm = a_col_sums
        if brs_max > 0:
            brs_norm = b_col_sums / brs_max
        else:
            brs_norm = b_col_sums
        C = -np.outer(ars_norm, brs_norm)

    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _surrogate_objective_1(a, b, p):
    """
    Surrogate 1: Linear flow-distance alignment.
    Minimizes sum_i (row_sum_a[i] * row_sum_b[p[i]])
    """
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    return np.dot(a_rs, b_rs[p])

def _surrogate_objective_2(a, b, p):
    """
    Surrogate 2: Quadratic penalty approximation.
    Uses a simplified interaction term to encourage spreading out high-flow facilities
    to low-distance locations.
    Approximation: sum_{i,j} a[i,j] * b[p[i], p[j]] is hard to compute fast.
    We use a proxy: sum_i (col_sum_a[i] * col_sum_b[p[i]]) 
    This captures different structural properties than row sums.
    """
    a_cs = a.sum(axis=0)
    b_cs = b.sum(axis=0)
    return np.dot(a_cs, b_cs[p])

def _dominates(obj1, obj2):
    """
    Check if solution 1 dominates solution 2.
    obj1 and obj2 are tuples (obj1_val, obj2_val, cost).
    Lower is better for all.
    """
    # Dominates if <= in all and < in at least one
    if (obj1[0] <= obj2[0] and obj1[1] <= obj2[1] and obj1[2] <= obj2[2]) and \
       (obj1[0] < obj2[0] or obj1[1] < obj2[1] or obj1[2] < obj2[2]):
        return True
    return False

def _pareto_filter(solutions):
    """
    solutions: list of dicts with 'p', 'cost', 'obj1', 'obj2'
    Returns list of non-dominated solutions.
    """
    pareto = []
    for s in solutions:
        dominated = False
        for ps in pareto:
            if _dominates((ps['obj1'], ps['obj2'], ps['cost']), (s['obj1'], s['obj2'], s['cost'])):
                dominated = True
                break
        if not dominated:
            # Check if new solution dominates any existing
            new_pareto = []
            for ps in pareto:
                if not _dominates((s['obj1'], s['obj2'], s['cost']), (ps['obj1'], ps['obj2'], ps['cost'])):
                    new_pareto.append(ps)
            pareto = new_pareto
            pareto.append(s)
    return pareto

def _local_search(a, b, p, time_limit, max_iter=200):
    """First-improvement local search for fast convergence."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    improved = True
    iter_count = 0
    while improved and time.time() - start_time < time_limit and iter_count < max_iter:
        improved = False
        iter_count += 1
        # Randomize order of swaps to avoid bias
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i_idx in range(n):
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = indices[j_idx]
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
    
    return best_p, best_cost

def _crossover(p1, p2, alpha):
    """
    Order-based crossover with mutation.
    alpha: 0.0 to 1.0, controls mixing.
    """
    n = len(p1)
    # Select a contiguous segment from p1
    start = np.random.randint(0, n)
    length = np.random.randint(1, n)
    end = (start + length) % n
    
    child = np.full(n, -1, dtype=np.int64)
    
    # Fill segment from p1
    if start + length <= n:
        child[start:start+length] = p1[start:start+length]
    else:
        # Wrap around
        seg1 = p1[start:]
        seg2 = p1[:end]
        child[start:] = seg1
        child[:end] = seg2
    
    # Fill rest with p2 in order
    p2_set = set(p2.tolist())
    for i in range(n):
        if child[i] == -1:
            for val in p2:
                if val in p2_set:
                    child[i] = val
                    p2_set.remove(val)
                    break
    
    # Mutation: random swaps
    num_mut = np.random.randint(1, 5)
    for _ in range(num_mut):
        i, j = np.random.choice(n, 2, replace=False)
        child[i], child[j] = child[j], child[i]
        
    return child

def _sa_polish(a, b, p, time_limit, t0=30000.0, t_final=0.05):
    """
    Simulated annealing polish phase with vectorized delta.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 110.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    # Initialize Pareto front with diverse solutions
    pop_size = 20
    population = []
    
    # Add Hungarian solutions for different objectives
    p_h1 = _hungarian_initial(a, b, 'linear')
    p_h2 = _hungarian_initial(a, b, 'quadratic')
    
    for p_init in [p_h1, p_h2]:
        p_ls, cost_ls = _local_search(a, b, p_init.copy(), 2.0)
        obj1 = _surrogate_objective_1(a, b, p_ls)
        obj2 = _surrogate_objective_2(a, b, p_ls)
        population.append({
            'p': p_ls,
            'cost': cost_ls,
            'obj1': obj1,
            'obj2': obj2
        })
    
    # Add random solutions
    while len(population) < pop_size:
        p_rand = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_rand, 1.0)
        obj1 = _surrogate_objective_1(a, b, p_ls)
        obj2 = _surrogate_objective_2(a, b, p_ls)
        population.append({
            'p': p_ls,
            'cost': cost_ls,
            'obj1': obj1,
            'obj2': obj2
        })
    
    # Filter to Pareto front
    pareto_front = _pareto_filter(population)
    
    best_p = min(pareto_front, key=lambda x: x['cost'])['p']
    best_cost = min(pareto_front, key=lambda x: x['cost'])['cost']
    
    # Evolutionary loop
    while time.time() - start < deadline - 10:
        # Generate offspring
        new_population = []
        num_offspring = len(pareto_front)
        
        for _ in range(num_offspring):
            # Select parents from Pareto front (uniformly for diversity)
            parent1 = pareto_front[np.random.randint(len(pareto_front))]
            parent2 = pareto_front[np.random.randint(len(pareto_front))]
            
            # Crossover
            child_p = _crossover(parent1['p'], parent2['p'], 0.5)
            
            # Local search polish
            time_remaining = deadline - (time.time() - start) - 5
            ls_time = min(2.0, time_remaining * 0.2)
            if ls_time < 0.5:
                break
                
            child_p_ls, child_cost = _local_search(a, b, child_p, ls_time)
            
            # Calculate surrogates
            obj1 = _surrogate_objective_1(a, b, child_p_ls)
            obj2 = _surrogate_objective_2(a, b, child_p_ls)
            
            new_population.append({
                'p': child_p_ls,
                'cost': child_cost,
                'obj1': obj1,
                'obj2': obj2
            })
            
            # Update global best
            if child_cost < best_cost:
                best_cost = child_cost
                best_p = child_p_ls.copy()
        
        # Combine and filter
        combined = pareto_front + new_population
        pareto_front = _pareto_filter(combined)
        
        # Limit front size to avoid excessive computation, keeping most diverse
        if len(pareto_front) > 30:
            # Simple crowding distance approximation: keep solutions with unique surrogate values
            # For now, just truncate to best cost if too large, but prioritize diversity
            # Sort by cost and take top 30
            pareto_front.sort(key=lambda x: x['cost'])
            pareto_front = pareto_front[:30]
        
        # Final SA polish on best solution if time permits
        time_remaining = deadline - (time.time() - start)
        if time_remaining > 15 and len(pareto_front) > 0:
            # Pick the best cost solution from front
            best_in_front = min(pareto_front, key=lambda x: x['cost'])
            p_final, cost_final = _sa_polish(a, b, best_in_front['p'].copy(), time_remaining - 5, t0=50000.0)
            if cost_final < best_cost:
                best_p = p_final
                best_cost = cost_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
