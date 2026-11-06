"""Rewritten program: maximize cardinality of subset of [1,100] with no subset summing to 200."""
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # Greedy baseline: iterate descending, add x if it doesn't create a subset sum of 200
    # using the reachable sums of the current selection.
    def greedy_baseline():
        selected = []
        reachable = {0}
        for x in range(100, 0, -1):
            # Check if adding x would create sum 200
            # i.e., is there s in reachable such that s + x == 200?
            if (200 - x) not in reachable:
                selected.append(x)
                # Update reachable sums
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable |= new_sums
        return selected

    best = greedy_baseline()
    best_len = len(best)

    # Local search: try to swap out smaller elements for larger ones, or add more elements
    # by removing some that block them.
    # Strategy: repeatedly try to add the largest missing element; if it conflicts,
    # try removing the smallest selected element that conflicts, then re-add.
    
    def can_add(selected_set, reachable, x):
        """Check if x can be added without creating sum 200."""
        return (200 - x) not in reachable

    def update_reachable(reachable, x, add=True):
        if add:
            new_sums = {s + x for s in reachable if s + x <= 200}
            return reachable | new_sums
        else:
            # Recompute from scratch is expensive; for now, we'll recompute when needed
            pass
        return reachable

    # More efficient local search using bitset-like approach with a set
    # We'll do iterative improvement:
    # 1. Start with greedy solution
    # 2. For each element not in selected (in descending order), try to add it.
    #    If it conflicts, find the minimal set of elements to remove to allow it.
    #    Simplification: remove the smallest element that conflicts.
    
    def local_search(initial_selected, time_limit):
        selected = list(initial_selected)
        selected_set = set(selected)
        # Compute reachable sums
        reachable = {0}
        for x in selected:
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each missing element in descending order
            for x in range(100, 0, -1):
                if x in selected_set:
                    continue
                if (200 - x) in reachable:
                    # Conflict: need to remove some element
                    # Find the smallest selected element y such that removing y allows adding x
                    # Simplified: try removing the smallest selected element
                    # This is heuristic; may not be optimal but can improve
                    # Better: find which elements in selected contribute to the sum 200-x
                    # For now, try removing the smallest element
                    if not selected:
                        continue
                    # Try removing the smallest element
                    y = min(selected)
                    # Check if removing y resolves the conflict
                    # We need to recompute reachable without y
                    # This is expensive; let's do a simpler check
                    # Actually, let's just try: remove y, recompute reachable, check if x can be added
                    selected.remove(y)
                    selected_set.discard(y)
                    # Recompute reachable
                    new_reachable = {0}
                    for z in selected:
                        ns = {s + z for s in new_reachable if s + z <= 200}
                        new_reachable |= ns
                    
                    if (200 - x) not in new_reachable:
                        # We can add x
                        selected.append(x)
                        selected_set.add(x)
                        # Update reachable
                        ns = {s + x for s in new_reachable if s + x <= 200}
                        new_reachable |= ns
                        reachable = new_reachable
                        improved = True
                    else:
                        # Put y back
                        selected.append(y)
                        selected_set.add(y)
                        # Recompute reachable (back to original)
                        new_reachable = {0}
                        for z in selected:
                            ns = {s + z for s in new_reachable if s + z <= 200}
                            new_reachable |= ns
                        reachable = new_reachable
                else:
                    # No conflict, just add
                    selected.append(x)
                    selected_set.add(x)
                    ns = {s + x for s in reachable if s + x <= 200}
                    reachable |= ns
                    improved = True
        
        return selected

    # Run local search with time budget
    if time.time() < deadline - 10:
        ls_result = local_search(best, deadline - 10)
        if len(ls_result) > best_len:
            best = ls_result
            best_len = len(best)

    # Additional pass: try to add more elements after local search
    # Simple greedy on the remaining elements
    if time.time() < deadline - 5:
        selected_set = set(best)
        reachable = {0}
        for x in best:
            ns = {s + x for s in reachable if s + x <= 200}
            reachable |= ns
        
        for x in range(100, 0, -1):
            if x in selected_set:
                continue
            if (200 - x) not in reachable:
                best.append(x)
                selected_set.add(x)
                ns = {s + x for s in reachable if s + x <= 200}
                reachable |= ns

    return best

# EVOLVE-BLOCK-END
