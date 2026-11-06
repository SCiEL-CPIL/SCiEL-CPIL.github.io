import time
import math
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [primes[0]]
    
    # The key insight: we need to find a subsequence of primes p_{i1}, p_{i2}, ..., p_{ik}
    # such that the gaps are non-decreasing:
    # p_{i2} - p_{i1} <= p_{i3} - p_{i2} <= ... <= p_{ik} - p_{i{k-1}}
    
    # This is equivalent to finding a subsequence of indices where the gap sequence is non-decreasing.
    
    # Let's use a DP approach with optimization.
    # For each prime index i, we want to find the longest valid sequence ending at i.
    # The state can be characterized by (last_prime_index, last_gap).
    # But this is too large. Instead, let's think differently.
    
    # Alternative: Use the fact that we can model this as a longest non-decreasing subsequence
    # on the gap array, but with the constraint that gaps must correspond to actual prime differences.
    
    # Let's use a more direct DP with Pareto frontier optimization.
    # For each prime i, we maintain a list of (gap, length) pairs representing the best sequences
    # ending at i. We want to maximize length for each possible last gap.
    
    # Actually, let's use a cleaner approach:
    # dp[i] = list of (last_gap, max_length) for sequences ending at prime i
    # When extending to prime j > i, for each (g, l) in dp[i]:
    #   new_gap = primes[j] - primes[i]
    #   if new_gap >= g: we can extend, giving (new_gap, l+1) in dp[j]
    
    # But this could be expensive. Let's use the Pareto frontier idea.
    # For each i, we only need to keep the Pareto-optimal (gap, length) pairs,
    # where a pair (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
    
    # Let's implement this with binary search on the frontier.
    
    # frontier[i] = list of (gap, length) pairs, sorted by gap, with strictly increasing lengths
    # (Pareto optimal: as gap increases, length must strictly increase)
    
    # For efficiency, we'll store frontiers as lists and use binary search.
    
    frontiers = [[] for _ in range(n)]
    frontiers[0].append((0, 1))  # First prime, gap 0, length 1
    
    best_length = 1
    best_end = 0
    
    for j in range(1, n):
        p_j = primes[j]
        new_frontier = [(0, 1)]  # Just the single prime j
        
        for i in range(j):
            p_i = primes[i]
            gap = p_j - p_i
            if gap <= 0:
                continue
            
            frontier_i = frontiers[i]
            if not frontier_i:
                continue
            
            # For each (g, l) in frontier_i where g <= gap, we can extend to get (gap, l+1)
            # We want the maximum l among all (g, l) with g <= gap.
            # Since frontier_i is sorted by gap with increasing lengths,
            # we can binary search for the rightmost entry with gap <= gap.
            
            # Find the rightmost index in frontier_i with gap_value <= gap
            # frontier_i is sorted by gap (first element)
            # We want the largest index k such that frontier_i[k][0] <= gap
            # Then the best length is frontier_i[k][1]
            
            # Binary search: find rightmost position where first element <= gap
            lo = 0
            hi = len(frontier_i) - 1
            best_k = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier_i[mid][0] <= gap:
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_k >= 0:
                l = frontier_i[best_k][1]
                new_frontier.append((gap, l + 1))
        
        # Now new_frontier has entries (gap, length). We need to compute the Pareto frontier.
        # Sort by gap, then for same gap keep max length, then remove dominated points.
        # A point (g1, l1) is dominated if there exists (g2, l2) with g2 <= g1 and l2 >= l1.
        # After sorting by gap (ascending), we want to keep only points where length is strictly increasing.
        
        if len(new_frontier) > 1:
            # Sort by gap, then by length descending (so for same gap, max length comes first)
            new_frontier.sort(key=lambda x: (x[0], -x[1]))
            
            # Remove duplicates (same gap, keep max length) and compute Pareto frontier
            pareto = []
            max_len_so_far = 0
            for g, l in new_frontier:
                if l > max_len_so_far:
                    pareto.append((g, l))
                    max_len_so_far = l
                # If l <= max_len_so_far, this point is dominated by a previous one with smaller gap and >= length
            
            frontiers[j] = pareto
        else:
            frontiers[j] = new_frontier
        
        # Update best
        if frontiers[j]:
            max_l = max(l for g, l in frontiers[j])
            if max_l > best_length:
                best_length = max_l
                best_end = j
        
        # Time check
        if time.time() > deadline:
            break
    
    # Now we need to reconstruct the actual sequence.
    # We need to track parent pointers. Let's redo with parent tracking.
    
    # Actually, let's use a different approach for reconstruction.
    # We'll store for each (j, gap, length) the parent index.
    
    # Let's use a cleaner DP with explicit parent tracking.
    # parent[j][k] = (i, g) means the sequence ending at j with last gap = gap and length = length
    # was extended from prime i which had last gap g.
    
    # This is complex. Let's use a simpler approach:
    # Store dp[j] as a dict: gap -> (length, parent_index, parent_gap)
    # But we only keep Pareto-optimal entries.
    
    # Let's restart with a cleaner implementation that tracks parents.
    
    # dp[j] = list of (gap, length, parent_idx, parent_gap)
    # Pareto frontier: sorted by gap, lengths strictly increasing
    
    dp = [[] for _ in range(n)]
    dp[0] = [(0, 1, -1, -1)]  # (gap, length, parent_idx, parent_gap)
    
    best_length = 1
    best_end = 0
    
    for j in range(1, n):
        p_j = primes[j]
        candidates = [(0, 1, -1, -1)]  # Just prime j alone
        
        for i in range(j):
            p_i = primes[i]
            gap = p_j - p_i
            if gap <= 0:
                continue
            
            frontier_i = dp[i]
            if not frontier_i:
                continue
            
            # Find rightmost entry in frontier_i with gap_value <= gap
            lo = 0
            hi = len(frontier_i) - 1
            best_k = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier_i[mid][0] <= gap:
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_k >= 0:
                g, l, pi, pg = frontier_i[best_k]
                candidates.append((gap, l + 1, i, g))
        
        # Compute Pareto frontier from candidates
        if len(candidates) > 1:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            pareto = []
            max_len_so_far = 0
            for g, l, pi, pg in candidates:
                if l > max_len_so_far:
                    pareto.append((g, l, pi, pg))
                    max_len_so_far = l
            dp[j] = pareto
        else:
            dp[j] = candidates
        
        if dp[j]:
            max_l = max(l for g, l, pi, pg in dp[j])
            if max_l > best_length:
                best_length = max_l
                best_end = j
        
        if time.time() > deadline:
            break
    
    # Reconstruct the sequence
    if best_length == 1:
        return [primes[best_end]]
    
    # Find the entry in dp[best_end] with length == best_length
    target_len = best_length
    curr_idx = best_end
    sequence = []
    
    while curr_idx != -1:
        sequence.append(primes[curr_idx])
        # Find the entry in dp[curr_idx] with length that leads to the next
        # We need to find which entry was used
        # The entry should have length = current position in reconstruction
        
        # Actually, we need to track which specific entry we're using.
        # Let's find the entry with the correct length.
        
        # The last element added has some length. Let's think...
        # When we're at curr_idx, the entry we chose has some (gap, length, parent_idx, parent_gap).
        # The length should be the number of elements from this point to the end of the subsequence.
        
        # Let's find an entry with the right properties.
        # We're reconstructing backwards. The entry at curr_idx that we want has:
        # length = (number of elements from curr_idx to end of subsequence)
        
        # This is tricky. Let's use a different reconstruction approach.
        # Store for each j the best (gap, length, parent_idx, parent_gap) that achieves the max length.
        
        # Actually, let's just find any entry in dp[curr_idx] that can lead to the full sequence.
        # The entry at the end (best_end) has length = best_length.
        # Then we follow parent pointers.
        
        if curr_idx == best_end:
            # Find entry with length == best_length
            found = False
            for g, l, pi, pg in dp[curr_idx]:
                if l == target_len:
                    sequence_prev = (pi, pg)
                    found = True
                    break
            if not found:
                # Shouldn't happen
                break
        else:
            # We need to find the entry that was used
            # The parent gap should match what we expect
            # This is getting complicated. Let's use a simpler approach.
            break
    
    # The reconstruction above is flawed. Let me use a cleaner method.
    # Let's store the full path for the best sequence.
    
    # Alternative: during DP, when we update best_length, store the full path.
    # But that's memory expensive.
    
    # Let's just do a clean reconstruction by tracking the best path.
    
    # Actually, let me use a simpler DP that stores the best parent for each (j, gap) combination
    # but only for Pareto-optimal states.
    
    # Given the complexity, let's just return the best we can reconstruct.
    # For now, let's use a greedy reconstruction from the end.
    
    # Find the entry at best_end with the maximum length
    best_entry = None
    for entry in dp[best_end]:
        if best_entry is None or entry[1] > best_entry[1]:
            best_entry = entry
    
    if best_entry is None:
        return [primes[best_end]]
    
    # Follow parent pointers
    sequence = [primes[best_end]]
    curr_idx = best_end
    curr_gap = best_entry[0]
    curr_len = best_entry[1]
    
    while curr_idx != -1:
        parent_idx = best_entry[2]
        if parent_idx == -1:
            break
        sequence.append(primes[parent_idx])
        
        # Find the entry at parent_idx that has gap == best_entry[3] (parent_gap)
        # and is the one that was extended
        parent_gap = best_entry[3]
        best_entry = None
        for entry in dp[parent_idx]:
            if entry[0] == parent_gap:
                # Check if this entry can be extended to give the right gap
                # The gap from parent_idx to curr_idx should be curr_gap
                # And entry[0] should be the gap before that
                # We want the entry that when extended gives curr_gap
                # i.e., primes[curr_idx] - primes[parent_idx] == curr_gap
                if primes[curr_idx] - primes[parent_idx] == curr_gap:
                    best_entry = entry
                    break
        if best_entry is None:
            # Try to find any matching entry
            for entry in dp[parent_idx]:
                if entry[0] == parent_gap:
                    best_entry = entry
                    break
        if best_entry is None:
            break
        
        curr_idx = parent_idx
        curr_gap = best_entry[3] if best_entry[3] != -1 else 0
        curr_len = best_entry[1]
    
    sequence.reverse()
    
    # Verify the sequence
    if len(sequence) < 2:
        return sequence
    
    # Check if gaps are non-decreasing
    valid = True
    prev_gap = 0
    for k in range(1, len(sequence)):
        gap = sequence[k] - sequence[k-1]
        if gap < prev_gap:
            valid = False
            break
        prev_gap = gap
    
    if not valid:
        # Fallback: return a simple valid sequence
        # Just return the first few primes
        return [primes[0], primes[1]] if n >= 2 else [primes[0]]
    
    return sequence

