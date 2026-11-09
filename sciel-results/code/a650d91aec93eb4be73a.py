"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # 15s safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def check_triple_sums_unique(arr):
        """Check if all unordered triples (with replacement) have unique sums."""
        n = len(arr)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = arr[i] + arr[j] + arr[k]
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def build_greedy_ordering(order, max_size=MAX_SIZE):
        """Greedy: iterate through order, add element if triple sums remain unique."""
        selected = []
        triple_sums = set()
        
        for x in order:
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            
            # Check if x can be added
            # New triples involving x: (x,x,x), (x,x,y) for y in selected, (x,y,z) for y<=z in selected
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                # (x, x, y) for each y in selected
                for y in selected:
                    s = 2 * x + y
                    if s in triple_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, z) for y <= z in selected
                n_sel = len(selected)
                for i in range(n_sel):
                    for j in range(i, n_sel):
                        s = x + selected[i] + selected[j]
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    def local_improvement(arr, max_size=MAX_SIZE):
        """Try to improve by swapping elements or adding new ones."""
        if not arr:
            return arr
        
        arr = list(arr)
        current_size = len(arr)
        
        # Try to add more elements
        for x in range(1, N + 1):
            if len(arr) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            if x in arr:
                continue
            
            candidate = arr + [x]
            if check_triple_sums_unique(candidate):
                arr = candidate
                current_size += 1
        
        # Try swapping: replace one element with another to potentially allow more additions
        for _ in range(5):  # Limited swaps
            if time.time() > DEADLINE - 10:
                break
            improved = False
            for i in range(len(arr)):
                for x in range(1, N + 1, 100):  # Sparse search for swap candidates
                    if x in arr or x == arr[i]:
                        continue
                    candidate = arr[:i] + [x] + arr[i+1:]
                    candidate.sort()
                    if check_triple_sums_unique(candidate):
                        # Try to extend this candidate
                        extended = build_greedy_ordering(range(1, N+1), max_size)
                        # Actually, let's just try to add to candidate
                        for y in range(1, N + 1):
                            if len(candidate) >= max_size:
                                break
                            if y in candidate:
                                continue
                            test = candidate + [y]
                            if check_triple_sums_unique(test):
                                candidate = test
                        if len(candidate) > len(arr):
                            arr = candidate
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return arr

    # Strategy 1: Simple greedy from 1
    best = build_greedy_ordering(range(1, N + 1))
    best_size = len(best)
    
    if time.time() < DEADLINE - 15:
        # Strategy 2: Greedy with different starting points
        for start in [2, 3, 5, 10, 50, 100]:
            if time.time() > DEADLINE - 15:
                break
            order = list(range(start, N + 1)) + list(range(1, start))
            candidate = build_greedy_ordering(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    if time.time() < DEADLINE - 10:
        # Strategy 3: Local improvement
        improved = local_improvement(best)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Strategy 4: Try base-representation inspired ordering
    if time.time() < DEADLINE - 10:
        # Numbers with large gaps in base-3 representation
        def base3_gap_score(x):
            s = 0
            prev_digit = -1
            gap = 0
            while x > 0:
                d = x % 3
                if prev_digit >= 0:
                    gap += abs(d - prev_digit)
                prev_digit = d
                x //= 3
            return gap
        
        order = sorted(range(1, N + 1), key=lambda x: -base3_gap_score(x))
        candidate = build_greedy_ordering(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Strategy 5: Powers of 2 based ordering (exponential growth)
    if time.time() < DEADLINE - 5:
        # Start with powers of 2, then fill in
        powers = [2**i for i in range(1, 13) if 2**i <= N]
        rest = [x for x in range(1, N + 1) if x not in set(powers)]
        order = powers + rest
        candidate = build_greedy_ordering(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Final local improvement if time permits
    if time.time() < DEADLINE - 5:
        improved = local_improvement(best)
        if len(improved) > best_size:
            best = improved
    
    return best
# EVOLVE-BLOCK-END
