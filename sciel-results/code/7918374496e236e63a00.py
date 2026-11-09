"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Precompute reciprocals as Fractions
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    # Check if a set S is valid
    def is_valid(S):
        S_set = set(S)
        for a in S_set:
            target = recips[a]
            others = [b for b in S_set if b != a]
            if _subset_sum_exists_fast(others, target):
                return False
        return True
    
    def _subset_sum_exists_fast(elements, target):
        """Fast check: can any subset of reciprocals of elements sum to target?"""
        if not elements:
            return target == 0
        # Use bitset-like approach with set of achievable sums, but prune aggressively
        achievable = {Fraction(0)}
        for elem in elements:
            recip = recips[elem]
            new_sums = set()
            for s in achievable:
                ns = s + recip
                if ns <= target:
                    new_sums.add(ns)
            if target in new_sums:
                return True
            achievable = achievable | new_sums
            # Early termination: if max possible sum < target, stop
            max_possible = sum(recips[e] for e in elements)
            if max_possible < target:
                return False
        return target in achievable
    
    # Precompute for each element a, which subsets of other elements can sum to 1/a
    # This is expensive, so we'll compute on demand with caching
    
    # Greedy construction with multiple orderings
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for elem in order:
            if elem in selected_set:
                continue
            # Check if 1/elem can be formed by subset of selected
            if _subset_sum_exists_fast(selected, recips[elem]):
                continue
            # Check if adding elem creates conflicts for existing elements
            valid = True
            for a in selected:
                # 1/a could now be formed using elem
                target_without_elem = recips[a] - recips[elem]
                if target_without_elem > 0:
                    others_without_a = [b for b in selected if b != a]
                    if _subset_sum_exists_fast(others_without_a, target_without_elem):
                        valid = False
                        break
            if valid:
                selected.append(elem)
                selected_set.add(elem)
        return selected
    
    # Try multiple greedy orderings
    best = []
    
    # Descending order (largest first)
    best = greedy_construct(range(24, 0, -1))
    
    # Ascending order
    asc_result = greedy_construct(range(1, 25))
    if len(asc_result) > len(best):
        best = asc_result
    
    # Try starting with large elements, then adding small ones
    base_large = list(range(15, 25))  # 15-24
    for small in range(1, 15):
        if time.time() > deadline - 10:
            break
        candidate = base_large + [small]
        if is_valid(candidate):
            base_large = candidate
    if len(base_large) > len(best):
        best = base_large
    
    # Try different bases
    for start in range(12, 20):
        if time.time() > deadline - 10:
            break
        base = list(range(start, 25))
        for small in range(1, start):
            if time.time() > deadline - 10:
                break
            candidate = base + [small]
            if is_valid(candidate):
                base = candidate
        if len(base) > len(best):
            best = base
    
    # Local search: try to add remaining elements
    best_set = set(best)
    remaining = [x for x in range(1, 25) if x not in best_set]
    improved = list(best)
    for elem in remaining:
        if time.time() > deadline - 15:
            break
        candidate = improved + [elem]
        if is_valid(candidate):
            improved = candidate
            best_set.add(elem)
    if len(improved) > len(best):
        best = improved
    
    # Try swap-based improvements: remove one, add two
    if time.time() < deadline - 20:
        best_set = set(best)
        for a in list(best_set):
            if time.time() > deadline - 15:
                break
            reduced = [x for x in best if x != a]
            reduced_set = set(reduced)
            for b in range(1, 25):
                if time.time() > deadline - 15:
                    break
                if b in reduced_set:
                    continue
                for c in range(b+1, 25):
                    if time.time() > deadline - 15:
                        break
                    if c in reduced_set:
                        continue
                    candidate = reduced + [b, c]
                    if is_valid(candidate):
                        if len(candidate) > len(best):
                            best = candidate
                            best_set = set(best)
    
    # Backtracking search for larger sets
    if time.time() < deadline - 30 and len(best) < 20:
        # Try to find a larger set via backtracking
        def backtrack(remaining_elements, current_set, current_set_set):
            nonlocal best
            if time.time() > deadline - 5:
                return
            if len(current_set) > len(best):
                best = list(current_set)
            if not remaining_elements:
                return
            # Prune: if we can't beat best
            if len(current_set) + len(remaining_elements) <= len(best):
                return
            
            for i, elem in enumerate(remaining_elements):
                if time.time() > deadline - 5:
                    return
                # Check if elem can be added
                if _subset_sum_exists_fast(current_set, recips[elem]):
                    continue
                # Check conflicts with existing elements
                valid = True
                for a in current_set:
                    target_without_elem = recips[a] - recips[elem]
                    if target_without_elem > 0:
                        others_without_a = [b for b in current_set if b != a]
                        if _subset_sum_exists_fast(others_without_a, target_without_elem):
                            valid = False
                            break
                if valid:
                    new_set = current_set + [elem]
                    new_set_set = current_set_set | {elem}
                    backtrack(remaining_elements[i+1:], new_set, new_set_set)
        
        # Start backtracking from a good base
        base_for_bt = [x for x in range(18, 25)]  # 18-24
        base_set = set(base_for_bt)
        remaining_for_bt = [x for x in range(1, 18) if x not in base_set]
        backtrack(remaining_for_bt, base_for_bt, base_set)
    
    # Final polish: try all single additions to best
    if time.time() < deadline - 10:
        best_set = set(best)
        for elem in range(1, 25):
            if time.time() > deadline - 5:
                break
            if elem in best_set:
                continue
            candidate = best + [elem]
            if is_valid(candidate):
                best = candidate
                best_set.add(elem)
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return list(best)

# EVOLVE-BLOCK-END
