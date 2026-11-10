# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    deadline = time.time() + 580  # Safety margin before 600s limit

    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Precompute for each element, which other elements are forbidden with it
    # forbidden_with[i] = set of j such that (i-j) % m or (j-i) % m is in forbidden
    forbidden_with = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            forbidden_with[i].add(j1)
            forbidden_with[i].add(j2)

    def greedy(order):
        """Greedy: process candidates in given order, add if compatible with all selected."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if candidate in forbidden_with[s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    def greedy_min_degree(seed_val):
        """
        Greedy independent set construction using minimum-degree heuristic.
        At each step, pick the vertex with minimum degree in the remaining candidate set,
        with random tie-breaking controlled by seed_val.
        """
        rng = random.Random(seed_val)
        candidates = set(range(m))
        selected = set()
        
        while candidates:
            min_deg = m
            min_nodes = []
            for v in candidates:
                deg = len(forbidden_with[v] & candidates)
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            
            v = rng.choice(min_nodes)
            selected.add(v)
            candidates.discard(v)
            for u in forbidden_with[v]:
                if u in candidates:
                    candidates.discard(u)
                    
        return selected

    def local_search_improve(initial_set, seed_val):
        """
        Improve an initial independent set using local search:
        1. Try to add any vertex that doesn't conflict.
        2. Try swaps: remove one vertex, add another, if valid and doesn't reduce size.
        Uses simulated annealing-like acceptance for swaps to escape local optima.
        """
        rng = random.Random(seed_val)
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        
        # Phase 1: Greedy addition
        improved = True
        while improved:
            improved = False
            for v in range(m):
                if v in current:
                    continue
                can_add = True
                for u in current:
                    if v in forbidden_with[u]:
                        can_add = False
                        break
                if can_add:
                    current.add(v)
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    break  # Restart to check again
        
        # Phase 2: Swap-based local search with random restarts
        # We keep the best set and try to improve it via swaps
        # Since size is fixed, we just want to find a different configuration that might allow additions
        # But additions are already maximized in Phase 1.
        # So we try to swap to find a "better" local minimum? 
        # Actually, for maximum independent set, if we can't add, we might need to remove one to add two?
        # That's a 1-2 swap. Let's try that.
        
        # Try 1-2 swaps: remove u, add v and w
        # This is expensive O(m^3). Let's try random 1-1 swaps first to shuffle, then try additions.
        
        for _ in range(50):  # Limited iterations to save time
            if time.time() > deadline:
                break
            
            # Pick a random vertex to remove
            if not current:
                break
            u = rng.choice(list(current))
            
            # Try to find two vertices v, w that can be added if u is removed
            # This is complex. Let's stick to simpler: 
            # Remove u, try to add any single v. If we can, size stays same, but configuration changes.
            # Then try to add more.
            
            # Actually, a more effective heuristic for MIS is "k-exchange".
            # Let's try removing u and adding v (1-1 swap) if it allows a subsequent addition.
            # Or just random 1-1 swaps to escape local minima, then re-run greedy addition.
            
            new_set = set(current)
            new_set.discard(u)
            
            # Find a v not in new_set that is compatible with new_set
            candidates_to_add = [v for v in range(m) if v not in new_set and all(v not in forbidden_with[x] for x in new_set)]
            
            if not candidates_to_add:
                continue
                
            v = rng.choice(candidates_to_add)
            new_set.add(v)
            
            # Now try to add more to new_set
            improved = True
            while improved:
                improved = False
                for w in range(m):
                    if w in new_set:
                        continue
                    can_add = True
                    for x in new_set:
                        if w in forbidden_with[x]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(w)
                        improved = True
                        break
            
            if len(new_set) > len(current):
                current = new_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                # Even if size is same, accept with some probability to explore
                if rng.random() < 0.3:
                    current = new_set
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)

        return best

    # Start with deterministic greedy in natural order
    best = greedy(list(range(m)))
    best_size = len(best)

    # Run multiple heuristics
    rng = random.Random(42)
    attempts = 0
    
    # First, try a few greedy_min_degree runs
    for i in range(20):
        if time.time() > deadline:
            break
        result = greedy_min_degree(i)
        if len(result) > best_size:
            best = list(result)
            best_size = len(result)

    # Now, run local search on the best set found so far
    if time.time() < deadline - 10:
        improved_best = local_search_improve(best, 42)
        if len(improved_best) > best_size:
            best = list(improved_best)
            best_size = len(improved_best)

    # Continue with random greedy and local search improvements
    while time.time() < deadline - 5:
        # Alternate between random greedy and min-degree greedy
        if attempts % 2 == 0:
            order = list(range(m))
            rng.shuffle(order)
            result = greedy(order)
        else:
            result = greedy_min_degree(attempts)
        
        # Apply local search to the result if it's promising
        if len(result) >= best_size - 1:  # Only improve if close to best
            improved_result = local_search_improve(result, attempts)
            if len(improved_result) > best_size:
                best = list(improved_result)
                best_size = len(improved_result)
        
        attempts += 1

    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
