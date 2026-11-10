"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Returns True if element is the mean of some subset of others (size >= 2)."""
        n = len(others)
        if n < 2:
            return False
        target = element
        
        # For each k from 2 to n, check if there exist k elements from others
        # whose sum equals k * target
        # Use DP: dp[j] = set of achievable sums using exactly j elements
        # Optimize: early exit if target_sum > sum of largest k elements or < sum of smallest k elements
        
        # Sort others for quick bound checks
        sorted_others = sorted(others)
        
        for k in range(2, n + 1):
            target_sum = k * target
            
            # Quick bounds check
            min_possible = sum(sorted_others[:k])
            max_possible = sum(sorted_others[-k:])
            if target_sum < min_possible or target_sum > max_possible:
                continue
            
            # DP: achievable sums with exactly j elements
            # Use a set for each count
            sums_by_count = [set() for _ in range(k + 1)]
            sums_by_count[0].add(0)
            
            for val in sorted_others:
                for j in range(k, 0, -1):
                    if sums_by_count[j - 1]:
                        for s in list(sums_by_count[j - 1]):
                            new_s = s + val
                            if new_s <= target_sum:
                                sums_by_count[j].add(new_s)
            
            if target_sum in sums_by_count[k]:
                return True
        
        return False
    
    def can_add_element(element, current_set):
        """Check if element can be added to current_set without violating the constraint.
        Only need to check:
        1. Is element the mean of any subset of current_set?
        2. For each a in current_set, is a the mean of any subset of (current_set - {a}) + {element}?
           This can be simplified: a is mean of subset of (current_set - {a}) + {element}
           iff a is mean of subset of current_set - {a} (already checked) 
           OR a is mean of subset that includes element.
           If a is mean of subset S of (current_set - {a}) + {element} that includes element,
           then sum(S) = |S| * a, and element is in S.
           So sum(S - {element}) = |S| * a - element = (|S|-1) * a + (a - element)
           Let T = S - {element}, then |T| = |S| - 1 >= 1 (since |S| >= 2 and element is in S)
           sum(T) = (|T|+1) * a - element
           So we need to check if there exists a subset T of current_set - {a} with |T| >= 1
           such that sum(T) = (|T|+1) * a - element
           Equivalently, for each j >= 1, check if there exists a subset of size j from current_set - {a}
           with sum = (j+1) * a - element
        """
        # Check 1: is element the mean of any subset of current_set?
        if is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            # Check if a is mean of any subset of others_without_a + [element]
            # that includes element
            # This means: exists subset T of others_without_a with |T| >= 1 such that
            # sum(T) + element = (|T| + 1) * a
            # => sum(T) = (|T| + 1) * a - element
            # => sum(T) - |T| * a = a - element
            # For each j from 1 to len(others_without_a):
            #   check if there's a subset of size j from others_without_a with sum = (j+1)*a - element
            
            n = len(others_without_a)
            for j in range(1, n + 1):
                target_sum = (j + 1) * a - element
                if target_sum < 0:
                    continue
                
                # Check if there's a subset of size j from others_without_a with sum = target_sum
                # Use DP
                sums_by_count = [set() for _ in range(j + 1)]
                sums_by_count[0].add(0)
                
                for val in others_without_a:
                    for jj in range(j, 0, -1):
                        if sums_by_count[jj - 1]:
                            for s in list(sums_by_count[jj - 1]):
                                new_s = s + val
                                if new_s <= target_sum:
                                    sums_by_count[jj].add(new_s)
                
                if target_sum in sums_by_count[j]:
                    return False
        
        return True
    
    # Strategy: build a large valid set
    # Use a greedy approach with multiple orderings
    
    best_set = []
    
    # Try different orderings
    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        # Powers of 2 first (known to work well)
        [2**i for i in range(9) if 2**i <= 512] + [x for x in range(1, 513) if x not in [2**i for i in range(9) if 2**i <= 512]],
        # Odds first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        # Non-multiples of 3 first
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        # Large numbers first (they're less likely to be means of smaller numbers)
        list(range(512, 256, -1)) + list(range(256, 0, -1)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        selected = []
        for x in ordering:
            if time.time() > deadline - 10:
                break
            if x in selected:
                continue
            
            if can_add_element(x, selected):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Local improvement: try to add more elements or swap
    if best_set and time.time() < deadline - 15:
        improved = best_set[:]
        remaining = [x for x in range(1, 513) if x not in improved]
        
        # Try adding remaining elements
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if can_add_element(x, improved):
                improved.append(x)
        
        if len(improved) > len(best_set):
            best_set = improved[:]
        
        # Try swapping: remove one element, try to add multiple others
        if time.time() < deadline - 20:
            for i in range(len(best_set)):
                if time.time() > deadline - 10:
                    break
                without_i = [x for j, x in enumerate(best_set) if j != i]
                removed = best_set[i]
                remaining2 = [x for x in range(1, 513) if x not in without_i]
                
                # Try to add as many as possible from remaining2
                new_set = without_i[:]
                added_count = 0
                for x in remaining2:
                    if time.time() > deadline - 10:
                        break
                    if can_add_element(x, new_set):
                        new_set.append(x)
                        added_count += 1
                
                if len(new_set) > len(best_set):
                    best_set = new_set[:]
                    break
    
    return best_set
# EVOLVE-BLOCK-END
