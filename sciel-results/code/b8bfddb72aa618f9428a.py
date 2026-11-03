import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Generate candidates: weight-2 (±2e_i), weight-3, weight-4 vectors
    candidates = []
    norms_sq = []
    
    # Type 1: Weight 2: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 2: Weight 3: ±e_i ± e_j ± e_k
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Type 3: Weight 4: ±e_i ± e_j ± e_k ± e_l
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < max_norm_sq (which is 4)
    max_norm_sq = 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < max_norm_sq
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as sets for fast intersection
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
    
    # Conflict counts for heuristic ordering
    conflict_counts = np.array([len(neighbors[i]) for i in range(n_candidates)])
    
    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # GRASP: Greedy Randomized Adaptive Search Procedure
    # Construction: restricted candidate list based on degree
    # Local search: 1-for-1 swaps to improve independent set size
    
    def grasp_construct(alpha, rng_seed):
        """
        GRASP construction phase.
        alpha controls the size of the restricted candidate list.
        Returns a list of selected indices (an independent set).
        """
        random.seed(rng_seed)
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            # Compute degrees of available nodes within available subgraph
            # For speed, sample if large
            if len(available) > 500:
                sample = random.sample(list(available), min(500, len(available)))
            else:
                sample = list(available)
                
            # Compute degree for each candidate in sample
            degs = []
            for v in sample:
                deg = len(neighbors[v] & available)
                degs.append(deg)
            
            if not degs:
                break
                
            min_deg = min(degs)
            max_deg = max(degs)
            
            # Restricted candidate list: nodes with degree in [min_deg, min_deg + alpha*(max_deg - min_deg)]
            threshold = min_deg + alpha * (max_deg - min_deg)
            rcl = [v for v, d in zip(sample, degs) if d <= threshold]
            
            # Pick randomly from RCL
            if rcl:
                chosen = random.choice(rcl)
            else:
                chosen = sample[0]
                
            selected.append(chosen)
            available.discard(chosen)
            
            # Remove neighbors of chosen from available
            for n in neighbors[chosen]:
                if n in available:
                    available.discard(n)
                    
        return selected
    
    def local_search_1swap(initial_set, max_iters=2000):
        """
        Local search: try 1-for-1 swaps (remove one, add one) to increase set size.
        Actually for independent set, we want to maximize size.
        A swap that keeps it independent and increases size is beneficial.
        But typically 1-swap means: remove vertex u, add vertex v.
        Size stays same. Instead, we should try:
        - Add a vertex that doesn't conflict (increases size by 1)
        - Or remove a vertex and add two that don't conflict with remaining (net +1)
        
        Let's do: try to add any compatible vertex. If none, try 1-2 swaps.
        """
        selected = list(initial_set)
        selected_set = set(selected)
        
        # First, try to add any compatible unselected vertex
        improved = True
        while improved:
            improved = False
            for v in range(n_candidates):
                if v not in selected_set:
                    if not (neighbors[v] & selected_set):
                        selected.append(v)
                        selected_set.add(v)
                        improved = True
                        break
        
        # Now try 1-for-1 swaps that allow adding more
        # Actually, let's try: for each u in selected, for each v not in selected:
        #   if we remove u and add v, can we then add more?
        # This is expensive. Instead, let's do a simpler improvement:
        # Try to find a vertex w not in selected such that neighbors[w] ∩ selected is small.
        # If we remove all neighbors of w from selected and add w, net change = 1 - |N(w) ∩ S|
        # This is beneficial if |N(w) ∩ S| = 0 (already handled above) or we can compensate.
        
        # Let's do: try all pairs (remove u, add v) where v was conflicting with u
        # This is too slow. Let's stick with greedy add and a few random restarts.
        
        return selected
    
    def grasp_full(alpha, n_iter, seed_offset=0):
        """
        Run GRASP for n_iter iterations.
        """
        best = []
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            sol = grasp_construct(alpha, 42 + seed_offset + it)
            
            # Local improvement: add any compatible vertex
            sol_set = set(sol)
            improved = True
            while improved:
                improved = False
                for v in range(n_candidates):
                    if v not in sol_set:
                        if not (neighbors[v] & sol_set):
                            sol.append(v)
                            sol_set.add(v)
                            improved = True
                            break
            
            if len(sol) > len(best):
                best = sol.copy()
                
        return best
    
    # Run GRASP with different alpha values
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_iter_per_alpha = 50
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
        
        sol = grasp_full(alpha, n_iter_per_alpha, seed_offset=int(alpha * 1000))
        if len(sol) > len(best_selected):
            best_selected = sol.copy()
    
    # Additional: min-degree greedy with multiple random seeds
    def greedy_min_degree(seed_val):
        random.seed(seed_val)
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 5.0:
                break
            
            # Sample to find min-degree node
            if len(available) > 300:
                sample = random.sample(list(available), 300)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        # Add any compatible vertex
        sol_set = set(selected)
        improved = True
        while improved:
            improved = False
            for v in range(n_candidates):
                if v not in sol_set:
                    if not (neighbors[v] & sol_set):
                        selected.append(v)
                        sol_set.add(v)
                        improved = True
                        break
                        
        return selected
    
    # Run min-degree greedy with multiple seeds
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        sol = greedy_min_degree(1000 + trial)
        if len(sol) > len(best_selected):
            best_selected = sol.copy()
    
    # Final: try to add any remaining compatible vectors
    if best_selected:
        best_set = set(best_selected)
        for v in range(n_candidates):
            if v not in best_set:
                if not (neighbors[v] & best_set):
                    best_selected.append(v)
                    best_set.add(v)
    
    if not best_selected:
        # Fallback: return a single vector
        v = [0] * D
        v[0] = 2
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
