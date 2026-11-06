"""B_3[1] construction: fast greedy with early rejection and swap-based improvement."""
import time
import random

def construct():
    deadline = time.time() + 82.0
    N = 4096
    
    def can_add(x, selected, sums):
        """Check if x can be added to selected without violating B_3[1] property."""
        # 3x
        if 3 * x in sums:
            return False
        # 2x + s for s in selected
        for s in selected:
            if 2 * x + s in sums:
                return False
        # x + 2s for s in selected
        for s in selected:
            if x + 2 * s in sums:
                return False
        # x + s + t for s <= t in selected
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x(x, selected, sums):
        """Add x to selected and update sums."""
        selected.append(x)
        sums.add(3 * x)
        for s in selected[:-1]:
            sums.add(2 * x + s)
            sums.add(x + 2 * s)
        n = len(selected)
        for i in range(n - 1):
            s = selected[i]
            for j in range(i, n - 1):
                t = selected[j]
                sums.add(x + s + t)

    def remove_x(x, selected, sums):
        """Remove x from selected and update sums."""
        idx = selected.index(x)
        selected.pop(idx)
        # Remove all sums involving x
        to_remove = set()
        to_remove.add(3 * x)
        for s in selected:
            to_remove.add(2 * x + s)
            to_remove.add(x + 2 * s)
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                to_remove.add(x + s + t)
        sums.difference_update(to_remove)

    # Phase 1: Standard greedy from 1 to 4096
    selected = []
    sums = set()
    
    for x in range(1, N + 1):
        if time.time() > deadline - 10.0:
            break
        if can_add(x, selected, sums):
            add_x(x, selected, sums)
    
    best_selected = selected[:]
    best_sums = sums.copy()
    best_size = len(selected)
    
    # Phase 2: Try to improve via swaps - optimized version
    # Key insight: prioritize larger candidates that are more likely to fit
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sel_set = set(selected)
        # Sort candidates by size descending - larger values are often easier to place
        candidates = sorted([x for x in range(1, N + 1) if x not in sel_set], reverse=True)
        
        # Use a counter to track how many times each candidate has been tried
        cand_attempts = {c: 0 for c in candidates}
        max_attempts_per_cand = 3
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                orig = selected[i]
                # Try candidates in order, but skip those already tried too many times
                for idx, cand in enumerate(candidates):
                    if time.time() > deadline - 3.0:
                        break
                    if cand_attempts[cand] >= max_attempts_per_cand:
                        continue
                    # Remove orig, try adding cand
                    remove_x(orig, selected, sums)
                    if can_add(cand, selected, sums):
                        add_x(cand, selected, sums)
                        # Remove successful candidate from list
                        candidates.pop(idx)
                        improved = True
                        break
                    else:
                        # Restore and mark attempt
                        add_x(orig, selected, sums)
                        cand_attempts[cand] += 1
                        # If we've tried enough small candidates, move on
                        if cand < orig:  # Only try larger or equal
                            break
                # Re-sort after potential swap
                selected.sort()
        
        if len(selected) > best_size:
            best_selected = selected[:]
            best_sums = sums.copy()
            best_size = len(selected)
    
    # Phase 3: Try adding more elements - use randomized restarts with different seeds
    if time.time() < deadline - 2.0:
        # Multiple attempts with different random orderings
        for attempt in range(5):
            if time.time() > deadline - 2.0:
                break
            
            sel_set = set(best_selected)
            # Try from high to low, but with some randomization
            candidates = [x for x in range(N, 0, -1) if x not in sel_set]
            if attempt == 0:
                # First attempt: pure descending
                pass
            else:
                # Subsequent attempts: shuffle slightly
                random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 2.0:
                    break
                # Try adding x to best_selected
                temp_selected = best_selected[:]
                temp_sums = best_sums.copy()
                if can_add(x, temp_selected, temp_sums):
                    add_x(x, temp_selected, temp_sums)
                    if len(temp_selected) > best_size:
                        best_selected = temp_selected
                        best_sums = temp_sums
                        best_size = len(temp_selected)
                        break  # Restart with new best
    
    return best_selected[:80]
