import itertools
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    points = range(v)
    all_k_subsets = list(itertools.combinations(points, k))
    num_blocks = len(all_k_subsets)
    
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Build sparse matrix A
    rows = []
    cols = []
    data = []
    
    for b_idx, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            t_idx = t_subset_to_idx[t_sub]
            rows.append(t_idx)
            cols.append(b_idx)
            data.append(1)
            
    A = sp.csr_matrix((data, (rows, cols)), shape=(num_t_subsets, num_blocks))
    
    # ILP Formulation
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = np.ones(num_blocks)
        bounds = Bounds(lb=0, ub=1)
        constraints = LinearConstraint(A, lb=np.ones(num_t_subsets), ub=np.inf)
        integrality = np.ones(num_blocks)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            selected_indices = np.where(res.x > 0.5)[0]
            result = [list(all_k_subsets[i]) for i in selected_indices]
            return result
    except Exception:
        pass
        
    # Fallback: Greedy
    # Precompute coverage lists for speed
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            t_to_blocks[t_subset_to_idx[t_sub]].append(b_idx)
            
    uncovered = [True] * num_t_subsets
    gain = [0] * num_blocks
    for b in range(num_blocks):
        gain[b] = len(all_k_subsets[b]) # C(k,t)
        
    selected_indices = []
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
        best_b = gain.index(max_gain)
        selected_indices.append(best_b)
        for t_sub in itertools.combinations(all_k_subsets[best_b], t):
            t_idx = t_subset_to_idx[t_sub]
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_to_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    result = [list(all_k_subsets[i]) for i in selected_indices]
    return result
