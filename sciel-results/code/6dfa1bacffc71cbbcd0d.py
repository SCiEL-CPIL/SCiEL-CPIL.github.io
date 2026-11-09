"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Construct a linear triple system on 16 points minimizing the independence number.
    Uses a hybrid approach:
    1. Start with a known good seed (e.g., from literature or heuristic).
    2. Use local search to minimize the independence number.
    3. Use exact branch-and-bound for small subproblems to verify/improve.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Points 0..15
    n = 16
    
    # Helper: check if two triples intersect in at most 1 point
    def is_linear(t1, t2):
        return len(set(t1) & set(t2)) <= 1
    
    # Helper: check if a list of triples is linear
    def is_system_linear(triples):
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                if not is_linear(triples[i], triples[j]):
                    return False
        return True
    
    # Helper: compute independence number (largest set with no triple contained)
    # We use a branch-and-bound or greedy+exact for small sets.
    # For 16 points, we can do this efficiently.
    def independence_number(triples):
        # Precompute for each point, the set of triples containing it
        # Actually, we want the largest subset S of points such that no triple is fully contained in S.
        # This is equivalent to: find max |S| such that for all t in triples, t not subset of S.
        
        # Use a simple branch and bound on the 16 points.
        # State: (current_set, remaining_points)
        # Pruning: if len(current_set) + len(remaining_points) <= best, prune.
        
        best = 0
        # Sort points for consistent ordering
        points = list(range(n))
        
        # Precompute: for each triple, the set of points
        triple_sets = [set(t) for t in triples]
        
        def is_valid_subset(subset):
            """Check if no triple is fully contained in subset."""
            subset_set = set(subset)
            for ts in triple_sets:
                if ts.issubset(subset_set):
                    return False
            return True
        
        # Branch and bound
        def bb(current, remaining):
            nonlocal best
            if not remaining:
                if len(current) > best:
                    best = len(current)
                return
            
            # Pruning
            if len(current) + len(remaining) <= best:
                return
            
            # Try to add each remaining point
            for i in range(len(remaining)):
                p = remaining[i]
                new_current = current + [p]
                # Check validity
                if is_valid_subset(new_current):
                    new_remaining = remaining[i+1:]
                    bb(new_current, new_remaining)
                # Also branch on NOT adding p (implicitly handled by trying next)
                # But we need to ensure we explore both branches? 
                # Standard MIS B&B: branch on include or exclude.
                # Here, we are building an independent set in the hypergraph sense.
                # We can just iterate.
        
        # To speed up, we can use a greedy heuristic to get a good initial lower bound for best
        # Greedy: add points that don't complete a triple
        greedy_set = []
        remaining = list(range(n))
        while remaining:
            # Pick point that minimizes number of new triples completed? 
            # Or just pick first valid point.
            found = False
            for i in range(len(remaining)):
                p = remaining[i]
                test_set = greedy_set + [p]
                if is_valid_subset(test_set):
                    greedy_set.append(p)
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                break
        
        best = len(greedy_set)
        
        # Now run B&B to see if we can do better
        bb([], list(range(n)))
        
        return best

    # Initial construction: Start with a random linear system or a known one.
    # Let's start with a cyclic design or random.
    # A simple linear system: Steiner Triple System S(2,3,7) doesn't cover 16.
    # Let's generate a random linear system.
    
    def generate_random_system():
        triples = []
        # Try to add random triples until no more can be added or we have enough
        # Max number of triples in a linear system on 16 points?
        # Each triple uses 3 points. Pair count: C(16,2)=120. Each triple uses 3 pairs.
        # So max triples <= 120/3 = 40.
        
        all_triples = list(combinations(range(n), 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            # Check if t intersects all existing triples in at most 1 point
            ok = True
            for existing in triples:
                if len(set(t) & set(existing)) > 1:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
        
        return triples
    
    # Local search move: replace a triple with another one, or swap points.
    # Let's try: pick a triple, remove it, add a new triple that keeps linearity.
    # Or: pick two triples, modify them.
    
    def local_search(triples, time_limit):
        current = triples
        current_alpha = independence_number(current)
        best = current
        best_alpha = current_alpha
        
        while time.time() < time_limit:
            # Pick a random triple to remove
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)
            
            # Try to add a new triple
            # Find all triples that are linear with current
            candidates = []
            for t in combinations(range(n), 3):
                ok = True
                for existing in current:
                    if len(set(t) & set(existing)) > 1:
                        ok = False
                        break
                if ok:
                    candidates.append(list(t))
            
            if not candidates:
                # Put it back
                current.insert(idx, removed)
                continue
            
            # Try a few random candidates
            random.shuffle(candidates)
            improved = False
            for new_t in candidates[:10]:  # Limit tries
                current.append(new_t)
                alpha = independence_number(current)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best = current.copy()
                    improved = True
                    break
                else:
                    current.pop()
            
            if not improved:
                # Revert
                current.insert(idx, removed)
        
        return best, best_alpha
    
    # Generate initial system
    sys1 = generate_random_system()
    if not sys1:
        # Fallback: simple disjoint triples
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    # Run local search
    best_sys, best_alpha = local_search(sys1, deadline - 10)
    
    # Try a few more random starts if time permits
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        sys2 = generate_random_system()
        if not sys2:
            continue
        sys2, alpha2 = local_search(sys2, deadline - 10)
        if alpha2 < best_alpha:
            best_sys = sys2
            best_alpha = alpha2
    
    return best_sys
# EVOLVE-BLOCK-END
