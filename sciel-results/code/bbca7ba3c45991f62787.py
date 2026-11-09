import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def calculate_independence_number(system):
        """
        Fast calculation of independence number for n=16.
        Returns the size of the largest independent set.
        """
        if not system:
            return 16
            
        # Pre-compute bitmask for each triple
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
        
        points = list(range(16))
        max_size = 0
        
        # Backtracking search with pruning
        # We want to find the max size of a subset S such that no triple is fully contained in S.
        # This is equivalent to finding the maximum independent set in the 3-uniform hypergraph.
        
        def backtrack(mask, size, start_idx):
            nonlocal max_size
            if size > max_size:
                max_size = size
                
            # Pruning: if size + (16 - start_idx) <= max_size, we can't beat current max
            if size + (16 - start_idx) <= max_size:
                return
                
            for i in range(start_idx, 16):
                new_mask = mask | (1 << i)
                
                # Check if new_mask is independent
                # A mask is independent if it does not fully contain any triple mask.
                is_independent = True
                for tm in triple_masks:
                    if (new_mask & tm) == tm:
                        is_independent = False
                        break
                        
                if is_independent:
                    backtrack(new_mask, size + 1, i + 1)
                # If not independent, we skip this branch. Note: we don't necessarily prune 
                # all supersets containing i, because removing some other point might make it valid, 
                # but since we are building up by adding points, if adding i creates a conflict 
                # with existing points in mask, then any superset containing mask ∪ {i} will 
                # also contain that conflict. So we can safely skip.
                
        backtrack(0, 0, 0)
        return max_size

    def greedy_packing(seed):
        """
        Generate a random maximal linear packing.
        """
        random.seed(seed)
        points = list(range(16))
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)
        
        triples = []
        used_pairs = set()
        
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append([a, b, c])
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                
        return triples

    def local_search(system, time_limit):
        """
        Improve the system by swapping points in triples to reduce independence number.
        Uses a fast bitmask-based independence check.
        """
        current_system = [list(t) for t in system]
        current_score = calculate_independence_number(current_system)
        
        # Precompute pair usage for quick linearity check
        def get_used_pairs(sys):
            used = set()
            for a, b, c in sys:
                used.add((a, b) if a < b else (b, a))
                used.add((b, c) if b < c else (c, b))
                used.add((a, c) if a < c else (c, a))
            return used

        used_pairs = get_used_pairs(current_system)
        
        step = 0
        while time.time() < time_limit:
            step += 1
            
            # Randomly select a triple to modify
            idx = random.randint(0, len(current_system) - 1)
            t = current_system[idx]
            
            # Try replacing each point in t with a random other point
            for k in range(3):
                old_point = t[k]
                candidates = [p for p in range(16) if p not in t]
                random.shuffle(candidates)
                
                for new_point in candidates:
                    # Construct new triple
                    new_t = t.copy()
                    new_t[k] = new_point
                    
                    # Check linearity
                    a, b, c = t
                    old_pairs = [
                        (a, b) if a < b else (b, a),
                        (b, c) if b < c else (c, b),
                        (a, c) if a < c else (c, a)
                    ]
                    
                    a, b, c = new_t
                    new_pairs = [
                        (a, b) if a < b else (b, a),
                        (b, c) if b < c else (c, b),
                        (a, c) if a < c else (c, a)
                    ]
                    
                    # Temporarily remove old pairs
                    for p in old_pairs:
                        used_pairs.remove(p)
                        
                    valid = True
                    for p in new_pairs:
                        if p in used_pairs:
                            valid = False
                            break
                            
                    if valid:
                        # Add new pairs
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        # Update system
                        current_system[idx] = new_t
                        
                        # Evaluate
                        new_score = calculate_independence_number(current_system)
                        
                        if new_score < current_score:
                            current_score = new_score
                            break
                        else:
                            # Revert
                            for p in new_pairs:
                                used_pairs.remove(p)
                            for p in old_pairs:
                                used_pairs.add(p)
                            current_system[idx] = t
                    else:
                        # Revert removal of old pairs
                        for p in old_pairs:
                            used_pairs.add(p)
                            
                    if current_score < calculate_independence_number(system): # Check if improved from start of this step? No, just break if improved
                        break
                if current_score < calculate_independence_number(system):
                    break
            # If no improvement in a full pass of candidates, we could restart or break.
            # For now, just continue.
            
        return current_system, current_score

    best_system = None
    best_score = 16
    
    # Run multiple instances of: Greedy + Local Search
    num_runs = 200 # Increased runs due to faster evaluation
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial packing
        init_system = greedy_packing(i)
        
        # Local search
        # Time limit per local search: 0.5s (Total 100s for 200 runs)
        search_start = time.time()
        optimized_system, score = local_search(init_system, time_limit=search_start + 0.5)
        
        if score < best_score:
            best_score = score
            best_system = optimized_system
            
            if best_score <= 5:
                break

    if best_system is None:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_system

# EVOLVE-BLOCK-END
