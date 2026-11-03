import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)

    # Weight 4
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
    # Conflict if dist_sq[i, j] < 4 (since max norm squared is 4)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors for each node
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())

    def greedy_min_degree(available_set):
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        """
        available = set(available_set)
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Find vertex with min degree in available subgraph
            # Sample if large
            if len(available) > 200:
                sample = random.sample(list(available), 200)
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
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        return selected

    best_selected = []
    np.random.seed(42)
    random.seed(42)

    # Run multiple trials with different random seeds
    num_trials = 300
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        # Use different random seed for each trial
        random.seed(42 + trial)
        sel = greedy_min_degree(set(range(n_candidates)))
        if len(sel) > len(best_selected):
            best_selected = sel

    # If we have time, try to improve by adding more vectors
    # via random search with weight-2 vectors
    if time.time() < deadline - 10.0 and best_selected:
        # Convert selected to set for fast lookup
        selected_set = set(best_selected)
        max_norm_sq = 4  # since we have weight-4 vectors
        
        # Try to add weight-2 vectors
        random.seed(12345)
        added = 0
        attempts = 0
        max_attempts = 100000
        
        while time.time() < deadline - 5.0 and attempts < max_attempts:
            attempts += 1
            
            # Random weight-2 vector
            i, j = random.sample(range(D), 2)
            v = np.zeros(D, dtype=np.int64)
            v[i] = random.choice([1, -1])
            v[j] = random.choice([1, -1])
            
            n_sq = int(np.dot(v, v))  # = 2
            
            # Check against all selected
            ok = True
            for s_idx in best_selected:
                s = candidates[s_idx]
                diff = v - s
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < max_norm_sq:
                    ok = False
                    break
            
            if ok:
                # Add this vector - but we need to handle it differently
                # since it's not in the candidates array
                # For now, skip this and just use the greedy result
                pass

    if not best_selected:
        # Fallback: return a simple valid configuration
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
