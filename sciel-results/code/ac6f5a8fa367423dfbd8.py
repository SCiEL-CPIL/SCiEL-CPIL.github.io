# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy construction with multiple restarts and local search (add/swap/remove).
    """
    m = 205
    k = 2
    
    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Precompute for each residue, which other residues are forbidden to pair with
    # forbidden_pairs[i] = set of j such that (i-j) mod m in forbidden or (j-i) mod m in forbidden
    # Actually, since forbidden is symmetric? Let's check: if d is a square, is -d a square?
    # Not necessarily. But we need both (a-b) mod m and (b-a) mod m not in forbidden.
    # So for a candidate c and existing s, we check:
    #   (c - s) % m not in forbidden AND (s - c) % m not in forbidden
    
    # Precompute: for each residue r, the set of residues that conflict with it
    # conflict[r] = set of s such that (r-s)%m in forbidden or (s-r)%m in forbidden
    conflict = [set() for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[a].add(b)
                conflict[b].add(a)
    
    def is_valid_subset(subset):
        """Check if subset is a valid square-difference-free set."""
        sset = set(subset)
        if len(sset) != len(subset):
            return False
        for a in subset:
            for b in subset:
                if a == b:
                    continue
                d1 = (a - b) % m
                d2 = (b - a) % m
                if d1 in forbidden or d2 in forbidden:
                    return False
        return True
    
    def greedy_random(seed=None):
        """Randomized greedy construction."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        candidates = list(range(m))
        rng.shuffle(candidates)
        
        selected = []
        selected_set = set()
        
        for candidate in candidates:
            ok = True
            for s in selected:
                if candidate in conflict[s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        
        return selected
    
    def local_search(initial_subset, max_iters=500, seed=None):
        """Local search with add, swap, and remove moves."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        selected = list(initial_subset)
        selected_set = set(selected)
        best_size = len(selected)
        best_subset = list(selected)
        
        for it in range(max_iters):
            improved = False
            
            # Try adding a new element
            candidates_to_add = [x for x in range(m) if x not in selected_set]
            rng.shuffle(candidates_to_add)
            for c in candidates_to_add:
                ok = True
                for s in selected:
                    if c in conflict[s]:
                        ok = False
                        break
                if ok:
                    selected.append(c)
                    selected_set.add(c)
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_subset = list(selected)
                    break  # Add one at a time
            
            # Try swapping: remove one, add one
            if not improved:
                # Try removing one element and then adding a different one
                for i in range(len(selected)):
                    old_elem = selected[i]
                    selected.pop(i)
                    selected_set.discard(old_elem)
                    
                    # Now try to add a better element
                    candidates_to_add = [x for x in range(m) if x not in selected_set]
                    rng.shuffle(candidates_to_add)
                    added = False
                    for c in candidates_to_add:
                        ok = True
                        for s in selected:
                            if c in conflict[s]:
                                ok = False
                                break
                        if ok:
                            selected.append(c)
                            selected_set.add(c)
                            if len(selected) > best_size:
                                best_size = len(selected)
                                best_subset = list(selected)
                            added = True
                            break
                    
                    if not added:
                        # Restore
                        selected.insert(i, old_elem)
                        selected_set.add(old_elem)
                    elif len(selected) <= best_size:
                        # Keep the swap if it's same size or better, otherwise revert
                        # Actually, we already updated best if it was better.
                        # If same size, we might want to keep exploring, but for simplicity, revert if not improved
                        # Let's just keep it for now to allow further exploration
                        pass
        
        return best_subset
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580s out of 630s, leaving 50s safety margin
    
    # Run multiple randomized greedy + local search iterations
    best_subset = []
    best_size = 0
    
    num_restarts = 500  # Maximum number of restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        initial = greedy_random(seed=restart)
        
        if len(initial) > best_size:
            best_size = len(initial)
            best_subset = list(initial)
        
        # Local search
        improved = local_search(initial, max_iters=200, seed=restart * 1000 + 1)
        
        if len(improved) > best_size:
            best_size = len(improved)
            best_subset = list(improved)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Final validation
    if not best_subset or not is_valid_subset(best_subset):
        # Fallback to simple deterministic greedy
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                if candidate in conflict[s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        best_subset = selected
    
    return np.array(best_subset, dtype=np.int64)

# EVOLVE-BLOCK-END
