import time
from math import lcm
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (5s buffer)

    # LCM of 1..22 for exact integer representation (for verification)
    L = reduce(lcm, range(1, 23))
    
    # Precompute reciprocal as integer: L // i
    # v[i-1] corresponds to element i
    v_exact = [L // i for i in range(1, 23)]
    
    # Helper to compute all subset sums for a candidate set
    # Returns the set of sums or None if invalid
    def get_sums(candidate):
        sums = {0}
        for idx in candidate:
            val = v_exact[idx - 1]
            new_sums = {s + val for s in sums}
            if not sums.isdisjoint(new_sums):
                return None
            sums.update(new_sums)
        return sums

    # Initial best result
    # Start with a known good set or find one quickly
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_result)
    
    # Verify initial
    if get_sums(best_result) is None:
         # Fallback to simple greedy if initial fails
         best_result = []
         for i in range(1, 23):
             if get_sums(best_result + [i]) is not None:
                 best_result.append(i)
         best_size = len(best_result)
    
    current_set = set(best_result)
    current_sums = get_sums(list(current_set))
    
    all_indices = set(range(1, 23))
    
    while time.time() < deadline:
        # 1. Try to Add (Hill Climbing step)
        added = False
        unused = list(all_indices - current_set)
        random.shuffle(unused)
        for idx in unused:
            val = v_exact[idx - 1]
            # Check if adding idx causes collision
            # New sums would be current_sums U {s + val for s in current_sums}
            # Collision if any s + val is in current_sums
            # i.e., if current_sums intersects {s + val for s in current_sums}
            # This is equivalent to checking if (current_sums - val) intersects current_sums
            # Or simply generate new sums and check.
            # For efficiency, check intersection directly.
            # new_sums = {s + val for s in current_sums}
            # if current_sums.isdisjoint(new_sums):
            #    valid
            # 
            # Optimization: 
            # s + val = t  => s = t - val
            # So we check if any (t - val) is in current_sums for t in current_sums.
            # This is O(N^2) if done naively. 
            # Generating new_sums is O(N). Checking disjoint is O(N). Total O(N).
            new_sums = {s + val for s in current_sums}
            if current_sums.isdisjoint(new_sums):
                # Valid add
                current_set.add(idx)
                current_sums = current_sums.union(new_sums)
                added = True
                # Update best if improved
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_result = list(current_set)
                    if best_size == 22:
                        return best_result
                break # Restart the loop to try adding another immediately? 
                      # Or continue to try more adds in this iteration?
                      # Hill climbing usually takes one step.
        
        if added:
            continue # Loop again to try more adds
            
        # 2. If no add possible, we are in a local optimum (w.r.t size).
        # Try to Swap or Remove to escape.
        
        # Strategy: Try Swaps first.
        # Pick a random used element u and a random unused element v.
        # Check if swapping u with v keeps the set valid.
        # This is expensive to check directly without removing u first.
        
        # Alternative: Random Restart with Perturbation.
        # If we are stuck, we might be in a "bad" local optimum.
        # Let's try a "kick": Remove 1-2 random elements, then try to add back.
        
        # Simple Kick: Remove a random element from current_set.
        if current_set:
            # Remove random element
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            # Recompute sums
            current_sums = get_sums(list(current_set))
            # If recompute returns None (shouldn't happen as subset of valid set is valid), 
            # but wait, removing an element from a dissociated set always yields a dissociated set.
            # So current_sums is valid.
            
            # Now try to add elements back, possibly different ones.
            # This is effectively a "re-greedy" from the perturbed state.
            # But we loop back to the top, which tries to Add.
            # This creates a cycle: Stuck -> Remove -> Add -> Stuck -> Remove...
            # To avoid oscillation, we can track recent states or just rely on randomness.
            
            # To make it more powerful, let's try a Swap explicitly.
            # We need an efficient swap check.
            # Swap u (in set) with v (not in set).
            # New set = (Set - {u}) U {v}.
            # Sums(Set - {u}) is a subset of Sums(Set).
            # Let S' = Sums(Set - {u}).
            # Then we check if adding v to S' is valid.
            
            # Computing S' from S is not trivial without recomputation.
            # Given N is small (~16), recomputation is fast (~65k ops).
            # Let's do a randomized swap trial.
            
            if unused: # There are unused elements
                u = random.choice(list(current_set))
                v = random.choice(unused)
                
                # Simulate swap
                temp_set = list(current_set)
                temp_set.remove(u)
                temp_set.append(v)
                
                new_sums = get_sums(temp_set)
                if new_sums is not None:
                    # Accept swap
                    current_set = set(temp_set)
                    current_sums = new_sums
                    # Note: Size is same, but structure changed.
                    # This might allow a future Add to succeed.
                    # We don't update best_size here as size is same.
                    # But we break out of the "stuck" state.
                    pass # Loop continues to try Adds.

    return best_result
