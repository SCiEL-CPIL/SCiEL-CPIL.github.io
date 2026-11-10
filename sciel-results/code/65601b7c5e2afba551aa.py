"""Improved program for two Sidon sets with disjoint difference sets using random restarts and local search."""
import time
import random
from itertools import combinations, combinations_with_replacement

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero absolute differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _greedy_construct(order, deadline, current_best_size):
    """
    Greedy construction given an order of candidates.
    Returns (sets, size) or (None, 0) if timeout.
    """
    sets = [[], []]
    diffs = [set(), set()]
    
    for x in order:
        if time.time() > deadline:
            return None, 0
            
        for side in (0, 1):
            # Check if adding x to sets[side] is valid
            # 1. Sidon condition for sets[side] + [x]
            # 2. Difference disjointness with sets[1-side]
            
            # Optimized check:
            # New sums involving x: x + a for a in sets[side], and x+x
            # New diffs involving x: |x - a| for a in sets[side]
            
            s = sets[side]
            o = sets[1-side]
            
            # Check Sidon
            valid_sidon = True
            for a in s:
                # Check if x+a collides with existing sums
                # Existing sums are all pairs from s.
                # We need to ensure x+a is not in {u+v | u,v in s, u<=v}
                # And x+x is not in that set.
                # This is O(|s|^2) if we recompute, but we can maintain a sum set.
                # For simplicity in this restart-based approach, let's just check quickly.
                # Actually, maintaining sum sets is better for speed.
                pass
            
            # Let's use a faster check by maintaining sum sets in the outer loop?
            # No, _greedy_construct is called per restart.
            # Let's just do the O(n^2) check but with early exit.
            
            # Check Sidon for s + [x]
            # Sums in s are unique.
            # New sums: x+x, and x+a for a in s.
            # We need to ensure:
            # 1. x+x is not in sums(s)
            # 2. x+a is not in sums(s) for any a in s
            # 3. x+a != x+b for any a != b in s (always true)
            
            # Let's compute sums(s) once per side update? No, s changes.
            # Let's just iterate.
            
            # Precompute sums of s?
            # If |s| is small (max 64), |s|^2 is 4096. 
            # 256 candidates * 2 sides * 4096 ops = ~2M ops per restart.
            # 100 restarts = 200M ops. Might be too slow in Python.
            
            # Better: Maintain sum_set and diff_set in the greedy loop.
            # But we are restarting.
            
            # Let's try a slightly optimized check inside the loop.
            
            # Check 1: Sidon
            # We need to know if x+a is in {u+v for u,v in s, u<=v}
            # Let's just build the set of sums for s each time? 
            # Or maintain it. Since we are in a linear scan of `order`, we can maintain it.
            
            # I will refactor to maintain sum_sets and diff_sets in the greedy loop.
            
            # Re-implementing logic with maintained sets for speed.
            # This function is called inside construct(), so I'll pass the state?
            # No, this is a separate function. I'll just write it efficiently.
            
            # Actually, the previous seed program did:
            # candidate = sets[side] + [x]
            # sums = ...
            # differences = ...
            # This is O(n^2) per check.
            
            # Let's stick to the simple check but add a time check more frequently.
            
            cand = s + [x]
            # Check Sidon
            seen_sums = set()
            ok = True
            for i in range(len(cand)):
                for j in range(i, len(cand)):
                    val = cand[i] + cand[j]
                    if val in seen_sums:
                        ok = False
                        break
                    seen_sums.add(val)
                if not ok:
                    break
            
            if not ok:
                continue
                
            # Check Diffs
            # New diffs: |x - a| for a in s
            # Must be disjoint from diffs(o)
            # And new diffs must be unique among themselves? 
            # No, Sidon set definition only requires unique sums. 
            # Differences just need to be disjoint from the OTHER set's differences.
            # Wait, the task says "nonzero difference sets must be disjoint".
            # It does NOT say differences within a set must be unique.
            # A Sidon set can have repeated differences (e.g., {1, 2, 4, 8} has diffs 1,2,3,4,6,7).
            # Wait, {1, 2, 4} diffs: 1, 3, 2. All unique?
            # {1, 3, 4} diffs: 2, 3, 1.
            # {1, 4, 6} diffs: 3, 5, 2.
            # Actually, if sums are unique, differences don't have to be unique.
            # Example: {1, 2, 5}. Sums: 2, 3, 6, 4, 7, 10. Unique.
            # Diffs: 1, 4, 3. Unique.
            # Example: {1, 3, 4}. Sums: 2, 4, 5, 6, 7, 8. Unique.
            # Diffs: 2, 3, 1.
            
            # So we just need:
            # 1. Sums in A are unique.
            # 2. Sums in B are unique.
            # 3. Diffs(A) ∩ Diffs(B) = empty.
            
            # So for side `side`:
            # - Check Sidon (done above)
            # - Check that new diffs (x-a for a in s) do not intersect with diffs[1-side]
            
            # Compute new diffs
            new_diffs = set()
            for a in s:
                new_diffs.add(abs(x - a))
            
            if new_diffs & diffs[1-side]:
                continue
            
            # Valid! Add x to side
            sets[side] = cand
            # Update diffs[side]
            diffs[side] = diffs[side].union(new_diffs)
            # Note: We don't need to check if new_diffs has internal duplicates for validity, 
            # only intersection with the other set.
            
            # Update current size?
            # We break out of side loop to try next x
            break
            
    total_size = len(sets[0]) + len(sets[1])
    return sets, total_size

def _local_improve(sets, deadline):
    """
    Try to improve the sets by swapping elements or inserting new ones.
    Returns improved sets.
    """
    # This is a placeholder for now, focusing on the greedy restarts first.
    # If we have time, we can add simple 1-swap local search.
    # For now, just return as is to keep it simple and fast.
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_sets = [[], []]
    best_size = 0
    
    # Initial deterministic pass to get a baseline quickly
    # Using the order 1..256
    order = list(range(1, 257))
    sets, size = _greedy_construct(order, deadline, 0)
    if sets:
        best_sets = sets
        best_size = size
    
    # Random restarts
    # We want to try different orderings.
    # Strategy: Random shuffle, or random priority based on "density" or just random.
    
    num_restarts = 1000 # Will be limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Create a random order
        # Sometimes mixing low and high numbers is good.
        # Let's just shuffle.
        order = list(range(1, 257))
        random.shuffle(order)
        
        sets, size = _greedy_construct(order, deadline, best_size)
        
        if sets and size > best_size:
            best_sets = sets
            best_size = size
            
            # If we hit the theoretical max?
            # Max size for Sidon set in [1, N] is roughly sqrt(N).
            # For N=256, sqrt(256)=16. 
            # But we have TWO sets with disjoint diffs.
            # The bound is tighter.
            # If best_size >= 30, we are doing well.
            # Let's not stop early unless we are very close to some known bound.
            
    return best_sets

# EVOLVE-BLOCK-END
