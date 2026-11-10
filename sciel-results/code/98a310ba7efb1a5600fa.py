"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute compatibility matrix
    # comp[a][b] is True if lcm(a,b) > 256
    # We use a set of sets for fast lookup
    # Actually, let's use a 2D list or just a set of tuples for incompatibility
    # Since we need to check lcm > 256, it's easier to store incompatibility (lcm <= 256)
    # Incompatible pairs are those where lcm(a,b) <= 256
    
    incompatible = set()
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= 256:
                incompatible.add((a, b))
                incompatible.add((b, a))
    
    def is_compatible(a, b):
        return (a, b) not in incompatible
    
    def check_pairwise_compatible(subset):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not is_compatible(subset[i], subset[j]):
                    return False
        return True
    
    def reciprocal_sum(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total
    
    # Greedy construction: start with small numbers (high reciprocal value)
    # but ensure compatibility
    # Strategy: Sort candidates by 1/x (descending), i.e., x ascending.
    # Try to add each candidate if it is compatible with all already selected.
    
    def greedy_construct():
        selected = []
        # We want high reciprocal sum, so we prefer small numbers.
        # But small numbers have many incompatibilities (lcm with small numbers is often small).
        # Let's try a few different orders.
        
        # Order 1: Ascending (small to large)
        order1 = sorted(candidates, key=lambda x: x)
        # Order 2: Descending (large to small) - likely bad for sum
        # Order 3: Random
        # Order 4: By number of incompatibilities (fewest first)
        
        # Let's compute incompatibility count for each number
        inc_count = {}
        for x in candidates:
            count = 0
            for y in candidates:
                if x != y and (x, y) in incompatible:
                    count += 1
            inc_count[x] = count
        
        # Strategy A: Greedy by ascending value
        selected_a = []
        for x in order1:
            if all(is_compatible(x, y) for y in selected_a):
                selected_a.append(x)
        
        # Strategy B: Greedy by fewest incompatibilities
        order_b = sorted(candidates, key=lambda x: (inc_count[x], x))
        selected_b = []
        for x in order_b:
            if all(is_compatible(x, y) for y in selected_b):
                selected_b.append(x)
        
        # Strategy C: Greedy by descending value (to see if large numbers help)
        order_c = sorted(candidates, key=lambda x: -x)
        selected_c = []
        for x in order_c:
            if all(is_compatible(x, y) for y in selected_c):
                selected_c.append(x)
        
        # Strategy D: Randomized greedy with multiple restarts
        best_random = []
        best_random_sum = Fraction(0, 1)
        
        # Time check
        if time.time() - start_time > timeout - 10:
            candidates_sets = [selected_a, selected_b, selected_c]
            best = max(candidates_sets, key=reciprocal_sum)
            return list(best)
        
        # Do some randomized greedy
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() - start_time > timeout - 5:
                break
            random.shuffle(candidates)
            selected_r = []
            for x in candidates:
                if all(is_compatible(x, y) for y in selected_r):
                    selected_r.append(x)
            s = reciprocal_sum(selected_r)
            if s > best_random_sum:
                best_random_sum = s
                best_random = list(selected_r)
        
        candidates_sets = [selected_a, selected_b, selected_c, best_random]
        best = max(candidates_sets, key=reciprocal_sum)
        
        # Local search: try to swap elements to improve sum
        # Since we want to maximize sum, we want to replace larger elements with smaller ones
        # if possible, or remove elements that allow adding smaller ones.
        
        current_best = list(best)
        current_sum = reciprocal_sum(current_best)
        
        improved = True
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            # Try removing each element and then greedily adding smaller ones?
            # Or try swapping a large element with a smaller compatible one.
            
            # Let's try a simple local search:
            # For each element in current_best, try to replace it with a smaller element
            # that is compatible with all other elements in current_best.
            
            # Sort current_best
            current_best.sort()
            
            for i in range(len(current_best)):
                if time.time() - start_time > timeout - 2:
                    break
                old_elem = current_best[i]
                # Try replacing with any x < old_elem not in current_best
                # such that x is compatible with all j != i in current_best
                others = [current_best[j] for j in range(len(current_best)) if j != i]
                # Find smallest x < old_elem compatible with others
                # Actually, we want to maximize sum, so we want the smallest possible x.
                # But replacing old_elem with x changes sum by 1/x - 1/old_elem.
                # We want 1/x > 1/old_elem, so x < old_elem.
                
                # Iterate through candidates smaller than old_elem
                for x in range(2, old_elem):
                    if x in current_best:
                        continue
                    if all(is_compatible(x, y) for y in others):
                        # Swap
                        new_set = list(current_best)
                        new_set[i] = x
                        new_set.sort()
                        new_sum = reciprocal_sum(new_set)
                        if new_sum > current_sum:
                            current_best = new_set
                            current_sum = new_sum
                            improved = True
                            break # Restart search from beginning as set changed
                if improved:
                    break
            
            # Also try adding new elements if possible?
            # The greedy construction should have added all possible compatible elements
            # in the initial phase, but local search might have created space.
            # Let's try to add any candidate not in current_best
            for x in candidates:
                if x in current_best:
                    continue
                if all(is_compatible(x, y) for y in current_best):
                    current_best.append(x)
                    current_best.sort()
                    current_sum = reciprocal_sum(current_best)
                    improved = True
                    break
        
        return current_best

    result = greedy_construct()
    
    # Final validation
    if not check_pairwise_compatible(result):
        # Fallback to simple greedy
        selected = []
        for x in range(2, N + 1):
            if all(is_compatible(x, y) for y in selected):
                selected.append(x)
        return selected
    
    return result
# EVOLVE-BLOCK-END
