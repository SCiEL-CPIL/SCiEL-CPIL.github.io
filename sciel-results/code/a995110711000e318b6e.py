"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(1, N + 1))
    
    # Precompute conflict graph: 
    # conflict[i][j] is True if gcd(i+1, j+1) > 1
    # We use 0-based indices for candidates
    n_cand = len(candidates)
    conflict = [[False] * n_cand for _ in range(n_cand)]
    
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if gcd(candidates[i], candidates[j]) > 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Sort candidates in descending order of value to prioritize high sums in search
    # This helps the branch-and-bound find good solutions early
    order = sorted(range(n_cand), key=lambda x: candidates[x], reverse=True)
    
    # Map original index to position in sorted order? 
    # Actually, it's easier to just work with the sorted list directly.
    # Let's create a list of (value, original_idx) or just reorder the candidates list.
    # Reordering candidates list changes indices, so let's stick to a permutation.
    
    # Let's use a recursive DFS with branch and bound.
    # State: 
    #   current_set: list of indices selected
    #   current_sum: sum of values in current_set
    #   next_idx: index in the sorted order to consider next
    
    # To make the bound tighter, we can precompute the sum of all remaining values.
    # But a better bound is: current_sum + sum of all values in remaining candidates 
    # that are compatible with all selected candidates.
    
    # For efficiency, we represent the set of "blocked" indices.
    # If we select index `i`, all `j` such that `conflict[i][j]` is True become blocked.
    
    # Let's use a bitset or a boolean array for blocked/available.
    # Given N=256, a boolean array of size 256 is fast enough.
    
    best_sum = 0
    best_set = []
    
    # Precompute suffix sums of the sorted candidates for a quick upper bound?
    # sorted_candidates[k] is the value at position k.
    # suffix_sum[k] = sum(sorted_candidates[k:])
    sorted_vals = [candidates[i] for i in order]
    suffix_sum = [0] * (n_cand + 1)
    for i in range(n_cand - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_vals[i]
    
    def dfs(pos, current_sum, available_mask):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
        
        # Pruning: if current_sum + sum of all remaining possible values <= best_sum, return
        # A quick upper bound is current_sum + suffix_sum[pos]
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        # If we've considered all candidates
        if pos == n_cand:
            if current_sum > best_sum:
                best_sum = current_sum
                # We need to reconstruct the set. 
                # Storing the set at every node is expensive. 
                # Alternative: store the path.
                # Let's store the path in a global list or pass it around.
                # For N=256, the set size is small (~20-30 items).
                # Let's pass a tuple or list of selected indices.
            return
        
        # Option 1: Exclude the candidate at `pos`
        # Option 2: Include the candidate at `pos` if it is available
        
        # It's often better to try including first if it's high value?
        # Since we sorted descending, high values are first.
        
        idx = order[pos]
        
        # Try including idx
        if available_mask[idx]:
            # Create new available mask
            # We need to mask out all conflicts with idx
            # This is O(N) per step. N=256, depth ~256. 256*256 = 65k ops.
            # This might be too slow if we do this naively at every node.
            # However, the branching factor is reduced by conflicts.
            
            # Optimization: Precompute conflict masks?
            # Let's just iterate.
            
            # Check if we should even try including it.
            # If current_sum + sorted_vals[pos] + suffix_sum[pos+1] <= best_sum, skip include?
            # No, that's the same as the bound check above but slightly tighter?
            # The bound check at the start of dfs uses suffix_sum[pos] which includes sorted_vals[pos].
            # So if we pass the initial check, including it is worth trying.
            
            # Construct new mask
            new_mask = [True] * n_cand # This allocation is expensive.
            # Let's use a different approach for the mask.
            # Since N=256, we can use an integer bitset? 
            # Python ints are arbitrary precision. 
            # Let's use a Python int as a bitset.
            # Bit i is 1 if candidate i is available.
            
            # Let's refactor to use bitsets.
            # This is a significant change. Let's restart the inner logic in my head.
            
            # Actually, for N=256, a list of bools is fine if we avoid deep copying.
            # We can use a "taint" array or simply pass a frozenset of blocked items?
            # Frozenset intersection is fast in C.
            
            # Let's use a frozenset of *available* indices.
            # Initial available: frozenset(range(n_cand))
            # When selecting idx: new_available = available - conflicts[idx]
            # We need to precompute conflicts as sets.
            
            pass # Placeholder for logic switch
            
    # Refactoring to use bitsets for speed
    # Bit i corresponds to candidate index i (0 to 255)
    
    conflict_masks = [0] * n_cand
    for i in range(n_cand):
        mask = 0
        for j in range(n_cand):
            if i != j and conflict[i][j]:
                mask |= (1 << j)
        conflict_masks[i] = mask
    
    full_mask = (1 << n_cand) - 1
    
    def dfs_bitset(pos, current_sum, available_bits):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of values of all bits set in available_bits 
        # that have index >= pos? 
        # Actually, `pos` is just the next one to consider in the sorted order.
        # The available_bits already reflects conflicts.
        # We can sum the values of all available bits.
        # But we must only consider bits with index >= pos?
        # No, the search is ordered. We only decide on items in order.
        # Items with index < pos have already been decided (either included or excluded).
        # So the remaining potential sum is the sum of values of available items with index >= pos.
        
        # Calculating this sum at every node is O(N). 
        # N=256. 
        # Let's precompute nothing for this, just calculate on the fly if needed?
        # Or use the suffix_sum approximation?
        # suffix_sum[pos] is the sum of ALL items from pos to end.
        # This is a valid upper bound (ignoring conflicts among remaining items).
        
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        if pos == n_cand:
            if current_sum > best_sum:
                best_sum = current_sum
                # Reconstruct set
                # We need to track the path. 
                # Let's pass a list of selected indices.
                # To avoid copying, we can use a global stack or pass a tuple.
                # Given the depth is small (max ~30-40 selected items), copying a list of size 40 is cheap.
                pass
        
        idx = order[pos]
        
        # Option 1: Exclude idx
        # The available bits for the next step are the same, except we don't need to worry about idx anymore.
        # But wait, `available_bits` represents which *future* candidates are still possible.
        # If we exclude idx, it doesn't block others.
        # So we just move to pos+1 with the same available_bits?
        # No, `available_bits` contains bits for ALL indices.
        # When we move to pos+1, we only care about bits >= pos+1?
        # Actually, the bitset tracks conflicts.
        # If we exclude idx, we don't remove its conflicts from the mask.
        # So we just call dfs_bitset(pos+1, current_sum, available_bits)
        
        dfs_bitset(pos + 1, current_sum, available_bits)
        
        # Option 2: Include idx
        if (available_bits >> idx) & 1:
            # Check if including it is promising
            # New available bits: remove idx and all its conflicts
            new_bits = available_bits & ~((1 << idx) | conflict_masks[idx])
            
            dfs_bitset(pos + 1, current_sum + candidates[idx], new_bits)
            
    # We need to track the best set. 
    # Let's modify the DFS to take a current_set list.
    
    best_set = []
    
    def dfs_with_set(pos, current_sum, available_bits, current_set):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
        
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        if pos == n_cand:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
        
        idx = order[pos]
        
        # Exclude
        dfs_with_set(pos + 1, current_sum, available_bits, current_set)
        
        # Include
        if (available_bits >> idx) & 1:
            new_bits = available_bits & ~((1 << idx) | conflict_masks[idx])
            current_set.append(idx)
            dfs_with_set(pos + 1, current_sum + candidates[idx], new_bits, current_set)
            current_set.pop()
            
    dfs_with_set(0, 0, full_mask, [])
    
    # Convert indices back to values
    result = [candidates[i] for i in best_set]
    
    # Ensure it's a list of ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
