"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected is sorted. sums_set contains all triple sums of selected.
        Returns (True, new_sums) if valid, else (False, None)."""
        n = len(selected)
        # New triple sums involving x:
        # 1. (x, x, x) = 3x
        # 2. (x, x, y) for y in selected -> 2x + y
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        
        # Check 3x
        if 3 * x in sums_set:
            return False, None
        
        new_sums = {3 * x}
        
        # Check 2x + y for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Check x + y + z for y <= z in selected
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums
    
    def greedy_forward(start, end, selected, sums_set):
        """Greedy forward from start to end."""
        for x in range(start, end + 1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    def greedy_backward(start, end, selected, sums_set):
        """Greedy backward from start down to end."""
        for x in range(start, end - 1, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    def greedy_random(selected, sums_set, candidates):
        """Greedy with random candidate selection."""
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    def rebuild_sums(selected):
        """Rebuild the sums set from scratch for a given sorted list."""
        sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(selected[i] + selected[j] + selected[k])
        return sums
    
    # Strategy 1: Greedy forward from 1 to 4096
    selected1 = []
    sums1 = set()
    selected1, sums1 = greedy_forward(1, 4096, selected1, sums1)
    best = selected1[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 2: Greedy backward from 4096 to 1
    selected2 = []
    sums2 = set()
    selected2, sums2 = greedy_backward(4096, 1, selected2, sums2)
    if len(selected2) > len(best):
        best = selected2[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 3: Greedy with randomization - try multiple random orders
    rng = random.Random(42)
    for trial in range(20):
        if time_left() < 5.0:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        selected3 = []
        sums3 = set()
        selected3, sums3 = greedy_random(selected3, sums3, candidates)
        if len(selected3) > len(best):
            best = selected3[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 4: Local search - try to swap elements to increase size
    # Start from best, try removing one element and adding a different one
    current = best[:]
    current_set = set(current)
    current_sums = rebuild_sums(current)
    
    improved = True
    while improved and time_left() > 10.0:
        improved = False
        # Try adding a new element
        # This is the most direct way to increase size
        for _ in range(200):
            if time_left() < 10.0:
                break
            x = rng.randint(1, 4096)
            if x in current_set:
                continue
            valid, new_sums = check_candidate(sorted(current), x, current_sums)
            if valid:
                current.append(x)
                current_set.add(x)
                current_sums.update(new_sums)
                improved = True
                if len(current) > len(best):
                    best = current[:]
        
        # Try swapping: remove one, add another
        # This can sometimes allow adding more elements later
        if time_left() > 15.0:
            for idx in range(len(current)):
                if time_left() < 15.0:
                    break
                old = current[idx]
                # Try a few random replacements
                for _ in range(10):
                    if time_left() < 15.0:
                        break
                    new = rng.randint(1, 4096)
                    if new in current_set or new == old:
                        continue
                    
                    # Temporarily remove old
                    test_list = current[:idx] + current[idx+1:]
                    test_set = current_set - {old}
                    test_sums = rebuild_sums(test_list)
                    
                    # Try to add new
                    valid, new_sums = check_candidate(sorted(test_list), new, test_sums)
                    if valid:
                        # Accept the swap if it doesn't reduce size (it keeps size same)
                        # But we only accept if it helps future additions? 
                        # For now, just accept if valid, as it might open up space
                        # Actually, swapping keeps size same. We only want to swap if it leads to improvement.
                        # Since we don't know, let's just do pure addition first.
                        # If pure addition stalls, then swap.
                        pass
    
    return best
# EVOLVE-BLOCK-END
