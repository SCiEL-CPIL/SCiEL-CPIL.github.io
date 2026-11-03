import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks for fast intersection checks
    masks = np.array([sum(1 << pos for pos in s) for s in subsets], dtype=np.int64)
    
    # Precompute adjacency: i and j conflict if they share exactly 3 elements
    # Use vectorized approach
    adj = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(i+1, n):
            inter = masks[i] & masks[j]
            if inter.bit_count() == 3:
                adj[i, j] = True
                adj[j, i] = True
    
    # Convert to list of sets for faster operations
    adj_sets = [set(np.where(adj[i])[0]) for i in range(n)]
    degrees = np.array([len(adj_sets[i]) for i in range(n)])
    
    def greedy_mis_dynamic(seed_val):
        """Dynamic greedy: always pick node with min current degree."""
        rng = random.Random(seed_val)
        available = set(range(n))
        is_set = set()
        # Track current degrees (degree within available set)
        cur_deg = {v: len(adj_sets[v] & available) for v in available}
        
        while available:
            # Pick node with minimum current degree, random tie-break
            min_deg = min(cur_deg.values())
            candidates = [v for v in available if cur_deg[v] == min_deg]
            best_node = rng.choice(candidates)
            
            is_set.add(best_node)
            available.discard(best_node)
            
            # Update degrees of remaining available nodes
            for nb in adj_sets[best_node]:
                if nb in available:
                    cur_deg[nb] -= 1
            
        return is_set
    
    def greedy_mis_random(seed_val):
        """Random order greedy."""
        rng = random.Random(seed_val)
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            if not (adj_sets[v] & is_set):
                is_set.add(v)
        return is_set
    
    def greedy_mis_weighted(seed_val):
        """Weighted greedy with random perturbation."""
        rng = random.Random(seed_val)
        available = set(range(n))
        is_set = set()
        cur_deg = {v: len(adj_sets[v] & available) for v in available}
        
        while available:
            # Score = current degree + small random perturbation
            best_node = None
            best_score = float('inf')
            for v in available:
                score = cur_deg[v] + rng.random() * 0.5
                if score < best_score:
                    best_score = score
                    best_node = v
            
            is_set.add(best_node)
            available.discard(best_node)
            
            for nb in adj_sets[best_node]:
                if nb in available:
                    cur_deg[nb] -= 1
        
        return is_set
    
    def local_search_2swap(is_set, rng, max_iter=30):
        """2-swap local search: try removing one and adding two."""
        is_set = set(is_set)
        for _ in range(max_iter):
            if time.time() > deadline:
                return is_set
            improved = False
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                # Try to add as many as possible
                added = 0
                candidates = [u for u in range(n) if u not in trial]
                rng.shuffle(candidates)
                for u in candidates:
                    if not (adj_sets[u] & trial):
                        trial.add(u)
                        added += 1
                if added >= 2:  # Net gain of at least 1
                    is_set = trial
                    improved = True
                    break
            if not improved:
                break
        return is_set
    
    best_is = set()
    best_size = 0
    
    current_seed = 0
    while time.time() < deadline:
        # Alternate between different greedy strategies
        if current_seed % 4 == 0:
            is_set = greedy_mis_dynamic(current_seed)
        elif current_seed % 4 == 1:
            is_set = greedy_mis_weighted(current_seed)
        elif current_seed % 4 == 2:
            is_set = greedy_mis_random(current_seed)
        else:
            is_set = greedy_mis_dynamic(current_seed + 1000)
        
        # Apply 2-swap local search
        rng = random.Random(current_seed)
        is_set = local_search_2swap(is_set, rng, max_iter=15)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            if best_size > 160:
                # Found good solution, can reduce iterations
                pass
            
        current_seed += 1
    
    # Build vectors from best independent set
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
