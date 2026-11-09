"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 4096
    MAX_SIZE = 80

    def compute_triple_sums(selected):
        """Compute all unordered triple sums for a sorted list."""
        sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(selected[i] + selected[j] + selected[k])
        return sums

    def can_add(selected, x, triple_sums):
        """Check if x can be added to selected without creating duplicate triple sums."""
        # New triples: (x,x,x), (x,x,y) for y in selected, (x,y,z) for y<=z in selected
        s = 3 * x
        if s in triple_sums:
            return False
        for y in selected:
            s = 2 * x + y
            if s in triple_sums:
                return False
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                s = x + selected[i] + selected[j]
                if s in triple_sums:
                    return False
        return True

    def add_element(selected, x, triple_sums):
        """Add x to selected and update triple_sums."""
        selected.append(x)
        selected.sort()
        triple_sums.add(3 * x)
        for y in selected[:-1]:
            triple_sums.add(2 * x + y)
        for i in range(len(selected) - 1):
            for j in range(i, len(selected) - 1):
                triple_sums.add(x + selected[i] + selected[j])

    def build_greedy_ordered(start_from=1, max_size=MAX_SIZE):
        """Greedy: add numbers in ascending order."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 20:
                break
            if can_add(selected, x, triple_sums):
                add_element(selected, x, triple_sums)
        
        return selected

    def build_greedy_descending(start_from=N, max_size=MAX_SIZE):
        """Greedy: add numbers in descending order."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, 0, -1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 20:
                break
            if can_add(selected, x, triple_sums):
                add_element(selected, x, triple_sums)
        
        return selected

    def build_base3_scaled(scale=1, max_size=MAX_SIZE):
        """Build using scaled base-3 sequence, then try to fill gaps."""
        selected = []
        x = scale
        while x <= N and len(selected) < max_size:
            if time.time() > DEADLINE - 20:
                break
            if can_add(selected, x, set()):  # fresh check
                selected.append(x)
            x *= 3
        
        # Now try to fill in gaps with smaller numbers
        triple_sums = compute_triple_sums(selected)
        for x in range(1, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 20:
                break
            if x not in selected and can_add(selected, x, triple_sums):
                add_element(selected, x, triple_sums)
        
        return selected

    def local_search_improve(selected, max_size=MAX_SIZE, iterations=500):
        """Try to improve by removing one element and adding a different one."""
        selected = list(selected)
        triple_sums = compute_triple_sums(selected)
        best = list(selected)
        best_size = len(selected)
        
        for _ in range(iterations):
            if time.time() > DEADLINE - 15:
                break
            if len(selected) >= max_size:
                break
            
            # Try removing each element and adding a new one
            improved = False
            for idx in range(len(selected)):
                if time.time() > DEADLINE - 15:
                    break
                removed = selected[idx]
                remaining = selected[:idx] + selected[idx+1:]
                rem_sums = compute_triple_sums(remaining)
                
                # Try to add a new element
                for x in range(1, N + 1):
                    if x in remaining:
                        continue
                    if can_add(remaining, x, rem_sums):
                        new_selected = remaining + [x]
                        new_selected.sort()
                        if len(new_selected) > best_size:
                            best = list(new_selected)
                            best_size = len(new_selected)
                            selected = new_selected
                            triple_sums = compute_triple_sums(selected)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best

    # Phase 1: Try multiple construction strategies
    candidates = []
    
    # Standard greedy from 1
    if time.time() < DEADLINE - 40:
        c1 = build_greedy_ordered(start_from=1)
        candidates.append(c1)
    
    # Base-3 scaled constructions
    if time.time() < DEADLINE - 40:
        for scale in [1, 2, 3, 4, 5]:
            if time.time() > DEADLINE - 40:
                break
            c = build_base3_scaled(scale=scale)
            candidates.append(c)
    
    # Descending greedy
    if time.time() < DEADLINE - 40:
        c = build_greedy_descending()
        candidates.append(c)
    
    # Pick best candidate
    if not candidates:
        return []
    
    best = max(candidates, key=len)
    best_size = len(best)
    
    # Phase 2: Local search to improve
    if time.time() < DEADLINE - 15:
        improved = local_search_improve(best, iterations=200)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: Try more local search with different parameters
    if time.time() < DEADLINE - 10:
        for start in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if time.time() > DEADLINE - 10:
                break
            c = build_greedy_ordered(start_from=start)
            if len(c) > best_size:
                improved = local_search_improve(c, iterations=100)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
