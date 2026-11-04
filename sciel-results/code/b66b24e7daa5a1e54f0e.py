import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    D = 11
    
    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    # adj[i] is a set of j such that masks[i] & masks[j] has bit_count 3
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute adjacency as boolean matrix for vectorized ops in local search
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj[i]:
            adj_bool[i][j] = True
            adj_bool[j][i] = True

    best_size = 0
    best_set = set()
    rng = np.random.RandomState(42)

    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            # Remove node and its neighbors from remaining
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s

    def greedy_random():
        """Randomized greedy independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            candidates = list(remaining)
            degs = [(sum(1 for nb in adj[v] if nb in remaining), v) for v in candidates]
            degs.sort()
            
            min_deg = degs[0][0]
            min_nodes = [v for d, v in degs if d == min_deg]
            
            min_node = min_nodes[rng.randint(len(min_nodes))]
            
            s.add(min_node)
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s

    def local_search(init_set, max_iter=2000):
        """Local search with 1-swap and 2-swap moves."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        in_set = np.zeros(N_S, dtype=bool)
        for v in current_set:
            in_set[v] = True
        
        def compute_blocked(set_obj):
            if not set_obj:
                return np.zeros(N_S, dtype=bool)
            return np.any(adj_bool[in_set], axis=0)
        
        blocked = compute_blocked(current_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            improved = False
            
            if len(cand_indices) > 0:
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                in_set[v_add] = True
                blocked |= adj_bool[v_add]
                
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                    improved = True
                continue
            else:
                if not current_set:
                    break
                    
                v_remove = list(current_set)[rng.randint(len(current_set))]
                current_set.remove(v_remove)
                in_set[v_remove] = False
                blocked = compute_blocked(current_set)
                
                added_count = 0
                candidates_mask = ~(in_set | blocked)
                cand_indices = np.where(candidates_mask)[0]
                
                if len(cand_indices) > 0:
                    idx_order = rng.permutation(len(cand_indices))
                    
                    for idx in idx_order:
                        if time.time() > deadline - 5.0:
                            break
                        v_add = int(cand_indices[idx])
                        if not in_set[v_add] and not blocked[v_add]:
                            current_set.add(v_add)
                            in_set[v_add] = True
                            blocked |= adj_bool[v_add]
                            added_count += 1
                
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)

        return best_local

    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            # Remove node and its neighbors from remaining
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s

    def greedy_random():
        """Randomized greedy independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            # Pick a node with random priority, or just random
            # Better: pick node with min degree, break ties randomly
            candidates = list(remaining)
            # Sort by degree in remaining
            degs = [(sum(1 for nb in adj[v] if nb in remaining), v) for v in candidates]
            degs.sort()
            
            # Get all nodes with min degree
            min_deg = degs[0][0]
            min_nodes = [v for d, v in degs if d == min_deg]
            
            # Pick one randomly
            min_node = min_nodes[rng.randint(len(min_nodes))]
            
            s.add(min_node)
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s

    def local_search(init_set, max_iter=2000):
        """Local search with 1-swap and 2-swap moves."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        # Precompute in_set and blocked for speed
        in_set = np.zeros(N_S, dtype=bool)
        for v in current_set:
            in_set[v] = True
        
        # blocked[v] is True if v is in current_set OR v is adjacent to any node in current_set
        # We want candidates: not in_set AND not adjacent to any in_set
        # Actually, independent set condition: no two nodes in set are adjacent.
        # So candidates are nodes not in set, such that they are not adjacent to ANY node in set.
        
        def compute_blocked(set_obj):
            if not set_obj:
                return np.zeros(N_S, dtype=bool)
            # Vectorized: OR of rows of adj_bool for nodes in set_obj
            # adj_bool[in_set] gives rows for nodes in set.
            # np.any(..., axis=0) gives columns that have any 1.
            return np.any(adj_bool[in_set], axis=0)
        
        blocked = compute_blocked(current_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Find candidates
            # A node v is a candidate if:
            # 1. v is not in current_set
            # 2. v is not adjacent to any node in current_set
            # Note: adj_bool does not include self-loops.
            # So "not adjacent to any in current_set" means ~blocked[v] is True?
            # blocked[v] is True if v is adjacent to some u in current_set.
            # So candidates are ~in_set & ~blocked.
            
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            improved = False
            
            if len(cand_indices) > 0:
                # Try to add a random candidate
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                in_set[v_add] = True
                # Update blocked
                # New blocked = old_blocked OR adj_bool[v_add]
                blocked |= adj_bool[v_add]
                
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                    improved = True
                continue
            else:
                # Stuck: no node can be added.
                # Try 1-swap: remove one node, see if we can add more than 1?
                # Or just remove one and try to add one (net 0, but might lead to better state)
                # Better: Try to remove a node that "blocks" many candidates, and then add multiple?
                # Simple 1-swap: Remove v_remove, then try to add v_add.
                # If we can add v_add, size stays same. We need to find a swap that increases size?
                # In independent set, if we are stuck, we can't add any.
                # If we remove v, we unblock its neighbors.
                # We can then try to add any neighbor of v.
                # This keeps size same.
                # To increase size, we need to remove 1 and add 2? (2-swap)
                
                # Let's try 2-swap:
                # Remove v1, v2. Add u1, u2.
                # This is complex.
                
                # Alternative: Simulated Annealing style.
                # Accept a move that keeps size same or decreases slightly?
                # Let's try: Remove a random node from current_set.
                # Then try to add as many candidates as possible greedily?
                
                if not current_set:
                    break
                    
                v_remove = list(current_set)[rng.randint(len(current_set))]
                current_set.remove(v_remove)
                in_set[v_remove] = False
                
                # Recompute blocked
                # We need to recompute from scratch or update carefully.
                # Since we removed a node, blocked might decrease.
                # Simplest: recompute.
                blocked = compute_blocked(current_set)
                
                # Now try to add candidates greedily
                added_count = 0
                candidates_mask = ~(in_set | blocked)
                cand_indices = np.where(candidates_mask)[0]
                
                # Sort candidates by degree (ascending) to pick "safe" ones first
                if len(cand_indices) > 0:
                    # Get degrees of candidates in the *remaining* graph?
                    # Just pick random ones
                    # Shuffle
                    idx_order = rng.permutation(len(cand_indices))
                    
                    for idx in idx_order:
                        if time.time() > deadline - 5.0:
                            break
                        v_add = int(cand_indices[idx])
                        # Check if still valid (might have been blocked by previous adds in this loop)
                        if not in_set[v_add] and not blocked[v_add]:
                            current_set.add(v_add)
                            in_set[v_add] = True
                            blocked |= adj_bool[v_add]
                            added_count += 1
                
                # If we added more than 1, we improved.
                # If we added 1, size is same (removed 1, added 1).
                # If we added 0, size decreased.
                
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)

        return best_local

    # Run multiple restarts
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r % 10 == 0:
            init_set = greedy_min_degree()
        else:
            init_set = greedy_random()
        
        final_set = local_search(init_set, max_iter=100)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
            
    # 4. Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        # Fallback: just one vector
        v = np.zeros(D, dtype=np.int8)
        v[0] = 1
        return np.array([v])
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
