"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE = 58.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_sidon(selected):
        """Check if selected is a valid Sidon set (all pair sums distinct including diagonals)."""
        sums = set()
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(selected):
        """Check if selected is inclusion-maximal: no missing element can be added."""
        selected_set = set(selected)
        sums = set()
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if adding x would create a conflict
            conflict = False
            for a in selected:
                s = x + a
                if s in sums:
                    conflict = True
                    break
            if not conflict:
                if 2 * x in sums:
                    conflict = True
            if not conflict:
                return False
        return True
    
    def greedy_sidon(order):
        """Greedy construction given an ordering of elements."""
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    def try_remove_element(selected, x):
        """Try to remove x from selected and check if result is still maximal."""
        if x not in selected:
            return None
        new_selected = [a for a in selected if a != x]
        if not new_selected:
            return None
        if not is_valid_sidon(new_selected):
            return None
        if is_maximal(new_selected):
            return new_selected
        return None
    
    def local_search_minimize(selected, max_iters=500):
        """Local search: try removing elements to minimize size while maintaining maximality."""
        current = list(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 5:
            improved = False
            iters += 1
            # Try removing each element in random order
            indices = list(range(len(current)))
            random.shuffle(indices)
            for idx in indices:
                x = current[idx]
                new_sel = try_remove_element(current, x)
                if new_sel is not None:
                    current = new_sel
                    improved = True
                    break
        return current
    
    def greedy_with_target_size(target_size, order):
        """Greedy construction that stops early if we reach a good size, then tries to extend maximality."""
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
            if len(selected) >= target_size:
                break
        # Check if this partial set can be made maximal by adding more elements
        # Actually, we need a full maximal set, so let's just do full greedy
        return selected
    
    # Strategy: Multiple random restarts with different orderings
    # Use various heuristics for ordering
    best_result = None
    best_size = N + 1
    
    # First, do the standard ascending greedy as a baseline
    baseline = greedy_sidon(list(range(1, N + 1)))
    if is_valid_sidon(baseline) and is_maximal(baseline):
        best_result = baseline
        best_size = len(baseline)
    
    # Try different random orderings with local search
    max_restarts = 200
    restart = 0
    while restart < max_restarts and elapsed() < DEADLINE - 10:
        restart += 1
        
        # Generate random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy construction
        selected = greedy_sidon(order)
        if not selected:
            continue
        
        # Verify
        if not is_valid_sidon(selected):
            continue
        if not is_maximal(selected):
            continue
        
        current_size = len(selected)
        if current_size >= best_size:
            # Still try local search to see if we can improve
            pass
        
        # Local search to minimize
        minimized = local_search_minimize(selected, max_iters=300)
        if minimized is not None:
            final_size = len(minimized)
            if final_size < best_size:
                best_size = final_size
                best_result = minimized
                # If we've gotten quite small, we can be more aggressive
                if best_size <= 12:
                    max_restarts = restart + 50
    
    # Final attempt: try to improve best_result further
    if best_result is not None and elapsed() < DEADLINE - 5:
        improved = local_search_minimize(best_result, max_iters=1000)
        if improved is not None and len(improved) < best_size:
            best_result = improved
    
    # Fallback to baseline if nothing better
    if best_result is None:
        best_result = baseline
    
    return best_result
# EVOLVE-BLOCK-END
