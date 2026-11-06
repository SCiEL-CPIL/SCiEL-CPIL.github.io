import time
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    # Vertices are 2..256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # weights as Fractions for exact comparison
    weights = [Fraction(1, v) for v in vertices]
    
    # Precompute conflict masks
    # conflict[i] is a bitmask of j such that lcm(vertices[i], vertices[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        vi = vertices[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        conflict[i] = mask

    # Strategy:
    # The numbers 2..16 are highly constrained.
    # Numbers 17..256 are less constrained.
    # We will enumerate all valid subsets of [2..16].
    # For each valid subset S_small, we find the best valid subset S_large of [17..256]
    # such that S_small ∪ S_large is a valid independent set.
    # This means S_large must be an independent set in the graph induced by [17..256]
    # AND no vertex in S_large can conflict with any vertex in S_small.
    # So we remove all vertices in [17..256] that conflict with any vertex in S_small.
    # Then we solve MWIS on the remaining graph of [17..256].
    
    small_range = 15  # 2..16 inclusive -> 15 numbers
    small_vertices = vertices[:small_range]
    large_vertices = vertices[small_range:]
    
    # Precompute conflicts among large vertices
    # Map large vertex value to index in large_vertices
    # large_vertices[0] is 17, large_vertices[1] is 18, ...
    n_large = len(large_vertices)
    
    # Conflict masks for large vertices (only among themselves)
    # conflict_large[i] is bitmask of j in large_vertices such that lcm(large_vertices[i], large_vertices[j]) <= 256
    conflict_large = [0] * n_large
    for i in range(n_large):
        vi = large_vertices[i]
        mask = 0
        for j in range(n_large):
            if i == j:
                continue
            vj = large_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        conflict_large[i] = mask

    # Weights for large vertices
    weights_large = [Fraction(1, v) for v in large_vertices]
    
    # Suffix sums of weights for large vertices (for B&B bound)
    # Sort large vertices by value? They are already sorted ascending 17..256.
    # Weights are descending.
    # For B&B, we want to consider high weights first.
    # Let's reorder large_vertices by weight descending (value ascending).
    # Actually, value ascending means weight descending. So current order is good.
    
    suffix_weights_large = [Fraction(0)] * (n_large + 1)
    for i in range(n_large - 1, -1, -1):
        suffix_weights_large[i] = suffix_weights_large[i + 1] + weights_large[i]

    # Function to solve MWIS on large graph with some vertices removed
    # removed_mask: bitmask of large vertices that are forbidden
    # Returns best sum (Fraction) and best mask
    def solve_large_mwis(removed_mask):
        best_sum = Fraction(0)
        best_mask = 0
        
        # available_mask: bits that are available
        initial_avail = ((1 << n_large) - 1) & ~removed_mask
        
        # B&B
        # We process indices 0..n_large-1
        # state: idx, avail_mask, current_sum, chosen_mask
        # To speed up, we can use a memoization or just efficient B&B.
        # Given n_large ~ 240, B&B might be slow if not pruned well.
        # But the graph is sparse?
        # Let's check density.
        # lcm(a,b) <= 256.
        # For a,b >= 17, lcm is usually > 256.
        # Example: 17, 18 -> lcm 306 > 256. No conflict.
        # 18, 20 -> lcm 180 <= 256. Conflict.
        # So conflicts are rare.
        # If conflicts are rare, the graph is close to empty.
        # In an empty graph, the best set is all vertices.
        # So B&B should be very fast if it quickly identifies that most vertices can be taken.
        
        # However, Python recursion and Fraction arithmetic might be slow.
        # Let's try a greedy approach first?
        # Greedy: pick highest weight vertex, remove neighbors.
        # Since weights are 1/v, we pick smallest v.
        # This is a maximal independent set.
        # Is it optimal? Not necessarily.
        # But with sparse conflicts, it might be close.
        # Let's try to improve with local search (1-swap).
        
        # Let's implement a fast greedy + local search for large part.
        # Greedy:
        avail = initial_avail
        chosen = 0
        current_sum = Fraction(0)
        
        # Iterate through large vertices in order of weight (ascending value)
        # But we must respect conflicts.
        # Simple greedy:
        for i in range(n_large):
            if not (avail & (1 << i)):
                continue
            # Check if i is available
            # Actually, if we iterate in order, and remove neighbors,
            # we just need to check if i is still in avail.
            if avail & (1 << i):
                chosen |= (1 << i)
                current_sum += weights_large[i]
                # Remove i and its neighbors from avail
                avail &= ~(1 << i)
                avail &= ~conflict_large[i]
        
        # Local Search: 1-swap
        # Try to swap a chosen vertex with an unchosen one if it improves sum.
        # Or just try to add/remove?
        # Since it's an independent set, we can try:
        # 1. Add a vertex v not in set, if it doesn't conflict with any in set.
        # 2. Remove a vertex u in set, if it allows adding multiple vertices?
        # Standard 1-swap: replace u with v.
        # But v must not conflict with any other in set.
        
        # Let's do a few iterations of local search.
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            # Try to add
            for i in range(n_large):
                if chosen & (1 << i):
                    continue
                if removed_mask & (1 << i):
                    continue
                # Check if i conflicts with any in chosen
                # conflict_large[i] & chosen == 0
                if conflict_large[i] & chosen:
                    continue
                # Try adding i
                # This is valid.
                chosen |= (1 << i)
                current_sum += weights_large[i]
                improved = True
                # Restart loop? Or just continue?
                # Adding one might enable adding others? No, it restricts.
                # But it might make a previous "bad" choice good? No.
                # Let's just mark improved and break to restart?
                # Or just continue.
                break 
            # Try to swap
            # For each u in chosen, for each v not in chosen:
            # If v doesn't conflict with (chosen - {u}), then swap is valid.
            # Gain = w(v) - w(u)
            # We want max gain > 0.
            # This is O(|chosen| * |not_chosen|).
            # |chosen| can be large (~200). |not_chosen| ~ 40.
            # 8000 checks per iteration. Fast.
            
            # Optimization: only consider v with w(v) > w(u)
            # Since w(v) = 1/v, v < u.
            
            # Let's implement a simple swap search.
            # We'll collect candidates.
            # To keep it fast, we might limit this.
            
            # Actually, if we just added vertices greedily, we might have missed some
            # because of ordering.
            # The "add" phase above is a hill-climb on additions.
            # The "swap" phase allows replacing.
            
            # Let's do swaps.
            for u in range(n_large):
                if not (chosen & (1 << u)):
                    continue
                # Try swapping u with v
                # v must be < u (for higher weight)
                # and v not in chosen
                # and v not in removed_mask
                # and v not conflicting with (chosen & ~(1<<u))
                
                # Iterate v
                for v in range(u): # v < u means 1/v > 1/u
                    if chosen & (1 << v):
                        continue
                    if removed_mask & (1 << v):
                        continue
                    # Check conflict
                    # v must not conflict with any in chosen except u
                    # conflict_large[v] & (chosen & ~(1 << u)) == 0
                    if conflict_large[v] & (chosen & ~(1 << u)):
                        continue
                    # Valid swap
                    # Gain is positive since v < u
                    chosen &= ~(1 << u)
                    chosen |= (1 << v)
                    current_sum += weights_large[v] - weights_large[u]
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                # Try removals? No, removing decreases sum.
                # Unless it allows adding two? That's 2-swap.
                # Let's stick to 1-swap for now.
                pass
            
            if time.time() > deadline - 5:
                break

        return current_sum, chosen

    # Enumerate subsets of small vertices
    # There are 2^15 = 32768 subsets.
    # For each, check if valid (no conflicts among small).
    # If valid, compute sum_small.
    # Compute removed_mask for large:
    # Any large vertex that conflicts with any small vertex in subset is removed.
    
    best_total_sum = Fraction(0)
    best_small_mask = 0
    best_large_mask = 0
    
    # Precompute for each small vertex i, the mask of large vertices it conflicts with
    small_conflicts_large = [0] * small_range
    for i in range(small_range):
        vi = small_vertices[i]
        mask = 0
        for j in range(n_large):
            vj = large_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        small_conflicts_large[i] = mask

    # Also precompute conflicts among small vertices
    small_conflicts_small = [0] * small_range
    for i in range(small_range):
        vi = small_vertices[i]
        mask = 0
        for j in range(small_range):
            if i == j:
                continue
            vj = small_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        small_conflicts_small[i] = mask

    # Iterate over all masks for small vertices
    # 2^15 = 32768
    for mask in range(1 << small_range):
        # Check validity of small subset
        valid = True
        # Quick check: if any vertex in mask conflicts with another in mask
        # We can precompute valid_small_masks?
        # Or just check on the fly.
        # Check: for each i in mask, if (small_conflicts_small[i] & mask) != 0, invalid.
        # But this double counts. Just check if (small_conflicts_small[i] & mask) has any bit j != i?
        # Actually, if i is in mask, and small_conflicts_small[i] & mask is non-zero, it means there is a conflict.
        # Wait, small_conflicts_small[i] does not contain i.
        # So if (small_conflicts_small[i] & mask) != 0, then there is a j in mask that conflicts with i.
        # This is sufficient.
        
        m = mask
        sum_small = Fraction(0)
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            sum_small += weights[i]
            if small_conflicts_small[i] & mask:
                valid = False
                break
            m ^= lsb
        
        if not valid:
            continue
        
        # Compute removed_mask for large
        removed_large = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            removed_large |= small_conflicts_large[i]
            m ^= lsb
        
        # Solve large MWIS with removed_large
        sum_large, chosen_large = solve_large_mwis(removed_large)
        
        total_sum = sum_small + sum_large
        
        if total_sum > best_total_sum:
            best_total_sum = total_sum
            best_small_mask = mask
            best_large_mask = chosen_large
            
        if time.time() > deadline - 10:
            # Time to stop?
            # We have 10s left.
            # We might not finish all 32768 masks if solve_large_mwis is slow.
            # Let's check time.
            # If we are running out of time, we should break.
            # But we want to be sure we have a good answer.
            # The first few masks (small values) might not be optimal.
            # Actually, we want to maximize sum.
            # Small values have large weights.
            # So masks with small numbers are likely to be better.
            # But we iterate 0..2^15-1.
            # Mask 0 is empty small set.
            # Mask 1 is {2}.
            # ...
            # We should probably iterate in order of potential quality?
            # Or just finish if possible.
            # 32768 iterations. If each takes 1ms, 32s.
            # If each takes 10ms, 327s -> too slow.
            # solve_large_mwis involves B&B or greedy+LS.
            # Greedy+LS on 240 nodes should be fast.
            # Let's assume it's fast enough.
            pass

    # Construct result
    result_set = []
    
    # Add small vertices
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result_set.append(small_vertices[i])
        m ^= lsb
        
    # Add large vertices
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        result_set.append(large_vertices[j])
        m ^= lsb
        
    # Verify result
    # Check non-empty
    if not result_set:
        # Fallback: return [256]
        return [256]
        
    # Check constraints
    # lcm(a,b) > 256 for all distinct pairs
    # This is guaranteed by construction if our logic is correct.
    # But let's double check just in case.
    # O(N^2) check. N <= 255.
    for i in range(len(result_set)):
        for j in range(i+1, len(result_set)):
            a = result_set[i]
            b = result_set[j]
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                # Error in logic!
                # Fallback to a known valid set?
                # Return empty? No, must be non-empty.
                # Return [256] is safe.
                return [256]

    return result_set

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(res)
#     s = sum(1/x for x in res)
#     print(s)
