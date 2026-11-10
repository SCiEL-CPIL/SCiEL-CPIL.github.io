import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute triples as tuples and masks
    triple_masks = []
    for t in all_triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    # For each triple, precompute its 3 pairs
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
    
    # For each pair (u,v), list of triples containing that pair
    # This is useful for checking linearity when adding a triple
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            pair_to_triples.setdefault(p, []).append(i)
    
    # Fast independence number calculation
    # We need to find the largest subset of vertices with no triple entirely contained in it.
    # Equivalently, find the maximum size of an independent set in the 3-uniform hypergraph.
    # 
    # Strategy: Use a branch-and-bound with memoization on (remaining_vertices_mask, min_index).
    # But with 16 vertices, we can be smarter.
    # 
    # Key insight: A vertex set S is independent iff for every triple in our system, 
    # S does not contain all 3 vertices of that triple.
    # 
    # We can use the following: build the "conflict" structure where two vertices are 
    # "in conflict" if there's a triple containing both. But that's not quite right for 
    # 3-uniform.
    # 
    # Better: Use inclusion-exclusion or direct search.
    # For 16 vertices, we can enumerate subsets by size (descending) and check.
    # But that's C(16,8) + ... which is large.
    # 
    # Alternative: Use a maximum independent set algorithm on the hypergraph.
    # We can use the fact that a set S is independent iff for each triple {a,b,c},
    # S is not a superset of {a,b,c}.
    # 
    # Efficient check: Given a candidate set S (as bitmask), S is independent iff
    # for every triple mask T in our active set, (T & S) != T.
    # 
    # For the search, we use branch and bound:
    # - Order vertices by "constraint" (number of triples they participate in)
    # - Branch on including or excluding each vertex
    # - Prune when current_size + remaining_vertices <= best_found
    
    def calc_alpha(triple_indices):
        """Compute independence number of the hypergraph defined by triple_indices."""
        if not triple_indices:
            return 16
        
        # Build set of active triple masks
        active_masks = set()
        for i in triple_indices:
            active_masks.add(triple_masks[i])
        
        # For each vertex, count how many active triples contain it
        deg = [0] * 16
        for tm in active_masks:
            for v in range(16):
                if tm & (1 << v):
                    deg[v] += 1
        
        # Order vertices by degree (most constrained first)
        order = sorted(range(16), key=lambda v: -deg[v])
        
        best = [0]
        
        # Precompute for each vertex v, the list of active triple masks containing v
        v2triples = [[] for _ in range(16)]
        for tm in active_masks:
            for v in range(16):
                if tm & (1 << v):
                    v2triples[v].append(tm)
        
        # For fast validity check: when adding vertex v to current_mask,
        # check if any active triple containing v has its other two vertices in current_mask
        def is_valid_add(current_mask, v):
            for tm in v2triples[v]:
                others = tm & ~(1 << v)
                if (others & current_mask) == others:
                    return False
            return True
        
        # Branch and bound search
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 8:
                return
            
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == 16:
                return
            
            v = order[idx]
            
            # Upper bound: current_size + (16 - idx)
            if current_size + (16 - idx) <= best[0]:
                return
            
            # Option 1: don't include v
            search(idx + 1, current_mask, current_size)
            
            # Option 2: include v (if valid)
            if is_valid_add(current_mask, v):
                search(idx + 1, current_mask | (1 << v), current_size + 1)
        
        search(0, 0, 0)
        return best[0]
    
    # Greedy initialization
    def greedy_init(seed_val):
        """Generate a maximal linear triple system via greedy packing."""
        rng = random.Random(seed_val)
        used_pairs = set()
        current = []
        
        indices = list(range(n_triples))
        rng.shuffle(indices)
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                for p in pairs:
                    used_pairs.add(p)
                current.append(t_idx)
        
        return current
    
    # Local search with targeted independent set hitting
    def local_search(current, seed_val):
        """Hill climbing with simulated annealing to minimize alpha."""
        rng = random.Random(seed_val)
        cur_alpha = calc_alpha(current)
        best_local_alpha = cur_alpha
        best_local = list(current)
        
        used_pairs = set()
        for t_idx in current:
            for p in triple_pairs[t_idx]:
                used_pairs.add(p)
        
        temp = 1.5
        cooling = 0.997
        
        while time.time() < deadline - 10:
            # Pick a random triple to remove
            if not current:
                break
            rem_idx = rng.randint(0, len(current) - 1)
            t_rem = current[rem_idx]
            
            for p in triple_pairs[t_rem]:
                used_pairs.discard(p)
            current.pop(rem_idx)
            
            # Try to add a random triple
            t_add = rng.randint(0, n_triples - 1)
            pairs_add = triple_pairs[t_add]
            valid = True
            for p in pairs_add:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                for p in pairs_add:
                    used_pairs.add(p)
                current.append(t_add)
                
                new_alpha = calc_alpha(current)
                
                if new_alpha <= cur_alpha:
                    cur_alpha = new_alpha
                    if new_alpha < best_local_alpha:
                        best_local_alpha = new_alpha
                        best_local = list(current)
                else:
                    import math
                    prob = math.exp(-(new_alpha - cur_alpha) / max(temp, 0.01))
                    if rng.random() < prob:
                        cur_alpha = new_alpha
                    else:
                        current.pop()
                        for p in pairs_add:
                            used_pairs.discard(p)
                        current.insert(rem_idx, t_rem)
                        for p in triple_pairs[t_rem]:
                            used_pairs.add(p)
            else:
                current.insert(rem_idx, t_rem)
                for p in triple_pairs[t_rem]:
                    used_pairs.add(p)
            
            temp *= cooling
        
        return best_local, best_local_alpha
    
    best_alpha = 17
    best_solution = None
    
    # Multiple restarts
    for r in range(12):
        if time.time() > deadline - 15:
            break
        
        seed_val = 42 + r * 97
        current = greedy_init(seed_val)
        
        alpha = calc_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current)
        
        if best_alpha <= 3:
            break
        
        best_local, local_alpha = local_search(current, seed_val)
        if local_alpha < best_alpha:
            best_alpha = local_alpha
            best_solution = best_local
        
        if best_alpha <= 3:
            break
    
    if best_solution is None:
        best_solution = list(range(min(10, n_triples)))
    
    return [list(all_triples[i]) for i in best_solution]
