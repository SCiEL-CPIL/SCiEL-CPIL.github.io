"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Split items into two halves for Meet-in-the-Middle
    # We want to find subsets S such that sum(1/d for d in S) < 1, maximizing the sum.
    # This is equivalent to finding subsets A from first half and B from second half
    # such that sum(A) + sum(B) < 1.
    
    # To ensure we don't miss anything, we can enumerate all subsets of the first half,
    # sort them by sum, and for each subset of the second half, find the best matching
    # subset from the first half that keeps the total < 1.
    
    # However, 2^24 is about 16 million. Storing Fractions for 16M items is memory heavy and slow.
    # Let's optimize: 
    # 1. Enumerate all subsets of the first half (indices 0..23).
    # 2. Keep only those with sum < 1.
    # 3. Sort them by sum.
    # 4. Enumerate all subsets of the second half (indices 24..46).
    # 5. For each subset B with sum < 1, binary search in the sorted list of A's for the largest sum A such that A + B < 1.
    
    # 2^24 = 16,777,216. This is borderline for Python speed and memory.
    # Let's try a slightly smaller split or a different approach if this is too slow.
    # Actually, 2^23 is 8 million. Let's split 23/24.
    
    n1 = 23
    n2 = n - n1
    
    half1 = fracs[:n1]
    half2 = fracs[n1:]
    
    # Enumerate all subsets of half1
    # We will store (sum, mask) pairs where sum < 1
    # To speed up, we can use integer arithmetic if possible, but Fractions are exact.
    # Given the constraint of 90s, 16M iterations in Python might take ~10-20s if optimized, 
    # but Fractions are slow.
    
    # Alternative: Use floating point for search, verify with Fractions at the end?
    # The deficit is very small. Floating point might lose precision.
    # Let's stick to Fractions but optimize the loop.
    
    # Let's try a different structural change: 
    # The problem is a subset sum problem with target < 1.
    # Since the items are 1/d, the sums are dyadic-like but with odd denominators.
    
    # Let's try enumerating the smaller half (23 items) and storing in a list.
    # 2^23 = 8,388,608.
    # Storing 8M Fractions might use ~1GB+ RAM. This is risky.
    
    # Better strategy: 
    # Enumerate all subsets of the full set? No, 2^47 is impossible.
    
    # Let's refine the MITM:
    # Instead of storing all sums, we can sort the items in half1 by value descending.
    # This doesn't help MITM directly.
    
    # Let's try a heuristic that is much stronger than local search:
    # Simulated Annealing or Tabu Search with a very strong initial guess.
    # Or, simply, a smarter exhaustive search on a reduced problem.
    
    # Observation: The optimal sum is very close to 1.
    # The largest fractions are 1/2, 1/3, 1/4...
    # Any solution close to 1 must include 1/2 (since sum of 1/3..1/48 is approx ln(48)-ln(2) - 1/2? No.
    # Sum(1/d for d=2..48) is approx ln(48) + gamma - 1.
    # ln(48) ~ 3.87. Sum is > 3.
    # So we have plenty of room.
    
    # Let's use a "Greedy with Backtracking" or "Branch and Bound".
    # Sort by fraction descending.
    # Branch on include/exclude.
    # Prune if current sum + sum of all remaining < best_sum.
    # Prune if current sum >= 1.
    
    sorted_fracs = sorted(enumerate(fracs), key=lambda x: x[1], reverse=True)
    # We need to keep track of original indices.
    # sorted_fracs[i] = (orig_idx, frac_val)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of fracs from i to end in sorted order
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_fracs[i][1]
    
    best_sum = Fraction(0)
    best_indices = []
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(current_indices)
            
        if idx == n:
            return
            
        # Pruning: if current_sum + suffix_sum[idx] <= best_sum, no improvement possible
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Option 1: Exclude current item
        dfs(idx + 1, current_sum, current_indices)
        
        # Option 2: Include current item
        new_sum = current_sum + sorted_fracs[idx][1]
        if new_sum < 1:
            current_indices.append(sorted_fracs[idx][0])
            dfs(idx + 1, new_sum, current_indices)
            current_indices.pop()
            
    # Start DFS
    dfs(0, Fraction(0), [])
    
    return sorted(best_indices)
# EVOLVE-BLOCK-END
