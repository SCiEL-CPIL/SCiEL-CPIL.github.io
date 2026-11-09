"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
from functools import reduce
from itertools import combinations
import time

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: Randomized local search with conflict-directed kicks.
    We maintain the set of subset sums as (value, bitmask) pairs.
    This allows us to quickly identify which elements are involved in a collision
    and perform targeted swaps to escape local optima.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = {i: L // i for i in universe}
    
    # Bit masks for each index
    # index i corresponds to bit (i-1)
    bit_masks = {i: 1 << (i - 1) for i in universe}
    
    def build_sums(indices):
        """Build list of (sum_value, bitmask) for the given indices."""
        sums = [(0, 0)]
        for idx in indices:
            w = weights[idx]
            mask = bit_masks[idx]
            # Add w to all existing sums
            current_len = len(sums)
            for j in range(current_len):
                s_val, s_mask = sums[j]
                sums.append((s_val + w, s_mask | mask))
        return sums

    def try_add(sums, indices, x):
        """
        Try to add element x to the current state.
        Returns (success, new_sums, conflict_mask).
        If success is False, conflict_mask indicates elements involved in collision.
        """
        w = weights[x]
        mask_x = bit_masks[x]
        s_set = {s_val for s_val, _ in sums}
        
        new_sums = []
        for s_val, s_mask in sums:
            ns_val = s_val + w
            ns_mask = s_mask | mask_x
            if ns_val in s_set:
                # Conflict found
                # Find the mask of the existing sum
                # We need to search for it
                for e_val, e_mask in sums:
                    if e_val == ns_val:
                        return False, None, (s_mask | e_mask)
            new_sums.append((ns_val, ns_mask))
        
        # No conflict
        # New sums are the union of old and new
        # We return the full list
        return True, sums + new_sums, 0

    # Initialization: Greedy
    def greedy(order):
        S = []
        sums = [(0, 0)]
        for i in order:
            success, new_sums, _ = try_add(sums, S, i)
            if success:
                S.append(i)
                sums = new_sums
        return S, sums

    best_set = []
    best_size = 0
    best_sums = []
    
    # Try a few greedy initializations
    orders = [
        universe,
        universe[::-1],
    ]
    for _ in range(10):
        o = universe[:]
        random.shuffle(o)
        orders.append(o)
        
    for o in orders:
        S, sums = greedy(o)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_sums = sums[:]

    current_set = best_set[:]
    current_sums = best_sums[:]
    current_size = len(current_set)
    
    start_time = time.time()
    timeout = 55.0
    
    # Local Search
    while time.time() - start_time < timeout:
        if current_size == N:
            break
            
        candidates = [i for i in universe if i not in current_set]
        if not candidates:
            break
            
        x = random.choice(candidates)
        
        success, new_sums, conflict_mask = try_add(current_sums, current_set, x)
        
        if success:
            # Accept add
            current_set.append(x)
            current_sums = new_sums
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
                best_sums = current_sums[:]
        else:
            # Conflict. Perform a kick.
            # Pick a random element from the conflict mask
            # conflict_mask is a bitmask
            # Find indices in current_set that are in conflict_mask
            conflict_indices = [i for i in current_set if conflict_mask & bit_masks[i]]
            
            if conflict_indices:
                y = random.choice(conflict_indices)
                mask_y = bit_masks[y]
                
                # Backup
                old_set = current_set[:]
                old_sums = current_sums[:]
                
                # Remove y
                new_set = [i for i in current_set if i != y]
                # Filter sums to remove those containing y
                new_sums = [(v, m) for v, m in current_sums if not (m & mask_y)]
                
                # Try to add x to the reduced state
                success2, final_sums, _ = try_add(new_sums, new_set, x)
                
                if success2:
                    # Swap successful: removed y, added x
                    current_set = new_set + [x]
                    current_sums = final_sums
                    # Size remains the same
                    # We don't update best_size here, but we might have a "better" set
                    # that allows future growth.
                else:
                    # Swap failed, restore
                    current_set = old_set
                    current_sums = old_sums
                    
    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
