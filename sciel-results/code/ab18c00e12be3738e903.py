"""Improved program for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    DEADLINE = start_time + 50  # safety margin before 90s kill
    
    # Precompute gcd/lcm checks
    # Two numbers a, b have lcm(a,b) > 256 iff a*b/gcd(a,b) > 256
    # i.e., a*b > 256 * gcd(a,b)
    
    def lcm_check(a, b):
        return a * b > 256 * gcd(a, b)
    
    def is_valid_set(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if not lcm_check(s[i], s[j]):
                    return False
        return True
    
    def reciprocal_sum(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total
    
    # Strategy: The key insight is that numbers that are "lcm-incompatible" 
    # with many others should be avoided unless they contribute high reciprocal value.
    # Small numbers have high reciprocal value but are more likely to have small lcm with others.
    # 
    # Key observation: lcm(a,b) <= 256 means a*b/gcd(a,b) <= 256.
    # For a < b, lcm(a,b) >= b, so if b > 256, lcm is always > 256. But b is at most 256.
    # For a, b in [2,256], lcm(a,b) > 256 is the constraint.
    # 
    # Two numbers a <= b have lcm(a,b) <= 256 iff b/gcd(a,b) * a <= 256 iff b * (a/gcd(a,b)) <= 256.
    # Let a = g*a', b = g*b' with gcd(a',b')=1. Then lcm = g*a'*b'. We need g*a'*b' > 256.
    # 
    # A number a "blocks" number b if lcm(a,b) <= 256.
    # 
    # Let's find which pairs are incompatible:
    # For a given a, b is incompatible if b/gcd(a,b) <= 256/a, i.e., b <= 256/gcd(a,b) * ... 
    # More precisely: lcm(a,b) = a*b/gcd(a,b) <= 256.
    
    # Precompute compatibility matrix
    all_numbers = list(range(2, 257))
    n = len(all_numbers)
    
    # compat[i][j] = True if lcm_check(all_numbers[i], all_numbers[j])
    # We'll use bitsets for efficiency
    # incompatible[i] = set of indices j where lcm_check is False
    
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        a = all_numbers[i]
        for j in range(i+1, n):
            b = all_numbers[j]
            if not lcm_check(a, b):
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # Greedy approach: sort by reciprocal value (descending, i.e., number ascending)
    # but we want to maximize sum, so we want small numbers.
    # However, small numbers block many others.
    
    # Let's try a greedy approach: iterate numbers from smallest to largest,
    # include if compatible with all already included.
    # This is what the seed does. Let's see what it gives.
    
    greedy_set = []
    for i in range(n):
        a = all_numbers[i]
        if all(j not in incompatible[k] for k in range(len(greedy_set)) for j in [k]) if False else True:
            pass
    
    # Simpler greedy:
    greedy_set = []
    greedy_indices = []
    for i in range(n):
        compatible = True
        for k in greedy_indices:
            if i in incompatible[k]:
                compatible = False
                break
        if compatible:
            greedy_set.append(all_numbers[i])
            greedy_indices.append(i)
    
    greedy_score = reciprocal_sum(greedy_set)
    
    # Now try to improve via local search
    best_set = list(greedy_set)
    best_score = greedy_score
    
    def try_add(s, idx):
        """Try adding index idx to set s (list of indices). Returns (new_set, score_delta) or (None, 0)."""
        for k in s:
            if idx in incompatible[k]:
                return None, Fraction(0, 1)
        return s + [idx], Fraction(1, all_numbers[idx])
    
    def try_remove(s, pos):
        """Remove element at position pos. Returns (new_set, score_delta)."""
        idx = s[pos]
        return s[:pos] + s[pos+1:], Fraction(-1, all_numbers[idx])
    
    def try_swap(s, pos, new_idx):
        """Replace element at position pos with new_idx. Returns (new_set, score_delta) or (None, 0)."""
        old_idx = s[pos]
        if new_idx == old_idx:
            return None, Fraction(0, 1)
        if new_idx in s:
            return None, Fraction(0, 1)
        remaining = s[:pos] + s[pos+1:]
        for k in remaining:
            if new_idx in incompatible[k]:
                return None, Fraction(0, 1)
        return remaining + [new_idx], Fraction(1, all_numbers[new_idx]) - Fraction(1, all_numbers[old_idx])
    
    # Local search: try single additions, removals, swaps
    import random
    random.seed(42)
    
    improved = True
    while improved:
        improved = False
        
        # Try additions
        for idx in range(n):
            if idx in best_set:
                continue
            new_set, delta = try_add(best_set, idx)
            if new_set is not None and delta > 0:
                best_set = new_set
                best_score += delta
                improved = True
                break
        
        # Try swaps
        if not improved:
            found_swap = False
            for pos in range(len(best_set)):
                old_idx = best_set[pos]
                for new_idx in range(n):
                    if new_idx == old_idx or new_idx in best_set:
                        continue
                    new_set, delta = try_swap(best_set, pos, new_idx)
                    if new_set is not None and delta > 0:
                        best_set = new_set
                        best_score += delta
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
            # Try removals (if something is blocking more than it contributes)
            if not improved:
                for pos in range(len(best_set)):
                    old_idx = best_set[pos]
                    new_set, delta = try_remove(best_set, pos)
                    # After removal, try to add something
                    if new_set is not None:
                        for idx in range(n):
                            if idx in new_set:
                                continue
                            new_set2, delta2 = try_add(new_set, idx)
                            if new_set2 is not None and delta + delta2 > 0:
                                best_set = new_set2
                                best_score += delta + delta2
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > DEADLINE:
                        break
    
    # Final check: verify validity
    if not is_valid_set(best_set):
        # Fallback to greedy
        best_set = greedy_set
        best_score = greedy_score
    
    if not best_set:
        best_set = [256]
    
    return best_set
# EVOLVE-BLOCK-END
