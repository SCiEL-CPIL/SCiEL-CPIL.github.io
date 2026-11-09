"""
Strategy: Greedy construction + local search, exploiting the lcm(a,b) > 256 constraint.

Key insight: If we sort numbers in decreasing order and greedily add them, the lcm condition
is often automatically satisfied because large numbers tend to have large lcms. The tricky
case is when two numbers share a large gcd. We use a smart greedy that prioritizes
high reciprocal weight (small numbers) while maintaining feasibility, then refines with
local search.
"""
import time
from math import gcd
from fractions import Fraction
import sys

def construct():
    start = time.time()
    deadline = start + 52.0  # Safety margin
    
    # Numbers 2..256
    # Weight: 1/n, we want to maximize sum
    
    # Precompute gcd and lcm relationships
    # Two numbers a, b are compatible if lcm(a,b) > 256, i.e., a*b/gcd(a,b) > 256
    
    N = 255  # numbers 2 to 256
    nums = list(range(2, 257))  # nums[0]=2, nums[254]=256
    
    # Compatibility matrix: compat[i][j] = True if lcm(nums[i], nums[j]) > 256
    # We only need to check pairs where we might want to include both
    # Precompute for all pairs
    compat = [[True] * N for _ in range(N)]
    for i in range(N):
        compat[i][i] = True
        ai = nums[i]
        for j in range(i + 1, N):
            aj = nums[j]
            g = gcd(ai, aj)
            lcm = ai * aj // g
            c = (lcm > 256)
            compat[i][j] = c
            compat[j][i] = c
    
    def is_valid_set(indices):
        """Check if a set of indices is valid (all pairs have lcm > 256)."""
        idxs = sorted(indices)
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                if not compat[idxs[ii]][idxs[jj]]:
                    return False
        return True
    
    def compute_score(indices):
        """Compute exact reciprocal sum as Fraction."""
        if not indices:
            return Fraction(0)
        s = Fraction(0)
        for i in indices:
            s += Fraction(1, nums[i])
        return s
    
    # Greedy construction: try to build a set
    # Strategy 1: Greedy by weight (smallest numbers first), adding if compatible
    def greedy_by_weight():
        # Sort by weight descending (i.e., by number ascending)
        order = sorted(range(N), key=lambda i: nums[i])
        selected = []
        for i in order:
            # Check if adding i is compatible with all selected
            ok = True
            for j in selected:
                if not compat[i][j]:
                    ok = False
                    break
            if ok:
                selected.append(i)
        return selected
    
    # Strategy 2: Greedy by number descending (largest first) - often gives larger sets
    def greedy_large_first():
        order = sorted(range(N), key=lambda i: -nums[i])
        selected = []
        for i in order:
            ok = True
            for j in selected:
                if not compat[i][j]:
                    ok = False
                    break
            if ok:
                selected.append(i)
        return selected
    
    # Strategy 3: Mixed - include small numbers if possible, then fill with large
    def greedy_mixed():
        # Start with a few small numbers that are "safe"
        # A number n is safe to include if it's coprime to many others
        # Actually, let's try: include n if it's compatible with all already included
        # But prioritize small n first, and if a small n conflicts, skip it
        order = sorted(range(N), key=lambda i: nums[i])
        selected = []
        for i in order:
            ok = True
            for j in selected:
                if not compat[i][j]:
                    ok = False
                    break
            if ok:
                selected.append(i)
        return selected
    
    # Local search: try to swap out one element for another to improve score
    def local_search(initial):
        current = list(initial)
        current_score = compute_score(current)
        best = list(current)
        best_score = current_score
        
        # Precompute which elements conflict with which
        # For each element i in current, find elements not in current that are compatible with all others
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try adding an element not in current
            in_current = set(current)
            # Candidate additions: not in current, compatible with all in current
            candidates_add = []
            for i in range(N):
                if i in in_current:
                    continue
                ok = True
                for j in current:
                    if not compat[i][j]:
                        ok = False
                        break
                if ok:
                    candidates_add.append(i)
            
            # Sort candidates by weight descending (smallest number first)
            candidates_add.sort(key=lambda i: nums[i])
            
            for i in candidates_add:
                if time.time() > deadline:
                    break
                new_set = current + [i]
                new_score = current_score + Fraction(1, nums[i])
                if new_score > best_score:
                    best_score = new_score
                    best = list(new_set)
                    current = list(new_set)
                    current_score = new_score
                    in_current.add(i)
                    improved = True
                    break  # Restart with the new addition
        
        # Try swaps: remove one, add another
        if time.time() < deadline:
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                
                in_current = set(current)
                # For each element in current, try removing it and adding a better one
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    rem = current[i]
                    # Remove rem
                    temp_set = [x for x in current if x != rem]
                    in_temp = set(temp_set)
                    
                    # Find best element to add
                    best_add = None
                    best_add_score = None
                    for j in range(N):
                        if j in in_temp or j == rem:
                            continue
                        ok = True
                        for k in temp_set:
                            if not compat[j][k]:
                                ok = False
                                break
                        if ok:
                            s = compute_score(temp_set) + Fraction(1, nums[j])
                            if best_add_score is None or s > best_add_score:
                                best_add_score = s
                                best_add = j
                    
                    if best_add is not None:
                        new_score = compute_score(temp_set) + Fraction(1, nums[best_add])
                        if new_score > best_score:
                            best_score = new_score
                            best = temp_set + [best_add]
                            current = list(best)
                            improved = True
                            break
        
        return best, best_score
    
    # Run multiple strategies and take the best
    best_result = []
    best_val = Fraction(0)
    
    strategies = [greedy_by_weight(), greedy_large_first(), greedy_mixed()]
    
    # Also try starting from empty and adding the single best
    for s in strategies:
        if time.time() > deadline:
            break
        score = compute_score(s)
        if score > best_val:
            best_val = score
            best_result = list(s)
    
    # Local search from the best initial
    if best_result and time.time() < deadline:
        refined, ref_score = local_search(best_result)
        if ref_score > best_val:
            best_val = ref_score
            best_result = refined
    
    # Additional local search rounds with different perturbations
    if time.time() < deadline and best_result:
        # Try removing a random element and re-optimizing
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            perturbed = list(best_result)
            if len(perturbed) > 1:
                # Remove a random element
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
            refined, ref_score = local_search(perturbed)
            if ref_score > best_val:
                best_val = ref_score
                best_result = refined
    
    # Final local search polish
    if time.time() < deadline and best_result:
        refined, ref_score = local_search(best_result)
        if ref_score > best_val:
            best_val = ref_score
            best_result = refined
    
    # Verify
    if not is_valid_set(best_result):
        # Fallback: single best element
        best_result = [0]  # number 2
    
    # Convert indices to numbers
    result = [nums[i] for i in best_result]
    
    return result
