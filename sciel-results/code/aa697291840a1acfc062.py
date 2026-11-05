"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n_primes = len(primes)
    
    # 2. Dynamic Programming approach
    # We want to find the longest sequence of primes p_1 < p_2 < ... < p_k
    # such that gaps g_i = p_{i+1} - p_i are non-decreasing (g_i <= g_{i+1}).
    
    # Let dp[i] be the length of the longest valid sequence ending at prime index i.
    # To compute dp[i], we need to find the best j < i such that:
    # 1. primes[i] - primes[j] >= gap_{j} (where gap_{j} is the gap of the sequence ending at j)
    # 2. The sequence ending at j is valid.
    
    # However, simply storing the length isn't enough because the "last gap" matters for the next step.
    # Let's refine the state.
    # A state can be defined by (last_prime_index, last_gap).
    # But last_gap can be large.
    
    # Alternative perspective:
    # Let f(i) be the maximum length of a valid sequence ending at primes[i].
    # This is not sufficient because two sequences ending at primes[i] with different last gaps
    # have different potentials for extension.
    # However, for a fixed end prime primes[i], a smaller last gap is always better for future extensions
    # because it imposes a weaker constraint on the next gap.
    # Therefore, for each index i, we only need to keep track of the sequence with the MINIMUM last gap
    # among all sequences of the MAXIMUM length ending at i?
    # No, that's not quite right. A shorter sequence with a much smaller last gap might extend further
    # than a longer sequence with a larger last gap.
    
    # Let's use a more robust DP:
    # Let dp[i] be a list of pairs (length, min_last_gap) for sequences ending at primes[i].
    # Actually, we can simplify.
    # For each i, we want to know: what is the longest sequence ending at i?
    # And among those, what is the minimum last gap?
    # Is it possible that a non-maximum length sequence at i has a smaller last gap and allows a longer total sequence?
    # Yes. Example:
    # Sequence A ends at i with length 5, last gap 100.
    # Sequence B ends at i with length 4, last gap 2.
    # If the next prime j is at distance 50 from i, A cannot extend, but B can.
    # So we need to keep Pareto-optimal states (length, last_gap).
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2.
    # We maintain a set of non-dominated states for each i.
    
    # Given N=1229, and gaps are integers, the number of non-dominated states should be small.
    # Let's implement this.
    
    # states[i] is a list of tuples (length, last_gap)
    # We want to maximize length.
    
    states = [[] for _ in range(n_primes)]
    
    # Initialize: each prime is a sequence of length 1 with last_gap 0
    for i in range(n_primes):
        states[i].append((1, 0))
        
    best_len = 1
    best_idx = 0
    best_gap = 0
    
    for i in range(n_primes):
        p_i = primes[i]
        # For each state ending at i, try to extend to j > i?
        # No, standard DP iterates i from 0 to N-1 and looks back at j < i.
        # So for each i, we look at all j < i.
        # If we can extend a sequence ending at j to i, we update states[i].
        
        # To be efficient, we can iterate j from 0 to i-1.
        # For each state in states[j], check if it can extend to i.
        
        # Optimization: 
        # The condition is: gap = p_i - p_j >= last_gap_j
        # If this holds, we can form a new state at i with length = len_j + 1 and last_gap = gap.
        
        # We can prune states[j] first to remove dominated ones?
        # Or just process all and prune states[i] after processing all j.
        
        # Since N is small (1229), O(N^2 * K) where K is avg states per index is fine.
        
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            # Check each state in states[j]
            # We can optimize by noting that if last_gap_j > gap, we cannot extend.
            # So we only care about states in states[j] with last_gap <= gap.
            
            for len_j, gap_j in states[j]:
                if gap_j <= gap:
                    # We can extend
                    new_len = len_j + 1
                    new_gap = gap
                    # Add (new_len, new_gap) to states[i]
                    # But we should check if it's dominated by existing states in states[i]
                    # Or just add it and prune later.
                    # Adding and pruning is safer.
                    states[i].append((new_len, new_gap))
                    
                    if new_len > best_len:
                        best_len = new_len
                        best_idx = i
                        best_gap = new_gap
                    elif new_len == best_len and new_gap < best_gap:
                        # Keep track of the best gap for the best length?
                        # Actually, we just need the final sequence.
                        # We will reconstruct at the end.
                        pass
                        
        # Prune states[i] to keep only non-dominated states
        # Sort by length descending, then gap ascending.
        # A state (L, G) is dominated if there exists (L', G') with L' >= L and G' <= G.
        # If we sort by L desc, then for a fixed L, we want the min G.
        # Actually, we can just filter.
        
        # Efficient pruning:
        # 1. Sort states by length descending.
        # 2. Iterate, keeping track of min_gap_seen_so_far for higher/equal lengths?
        # No, standard Pareto front.
        # Sort by length descending.
        # Keep a list.
        # For each state, if its gap is smaller than all gaps of states with higher length, it's non-dominated?
        # No. (5, 10) and (4, 5). (5,10) is not dominated by (4,5) because 5>4.
        # (4,5) is not dominated by (5,10) because 5<10 but 4<5.
        # So we need to keep both?
        # Wait, if we have (5, 10) and (4, 5).
        # Can (4,5) lead to a longer total sequence?
        # Yes, if the next gap is 6. (4,5) extends to len 5 gap 6. (5,10) cannot.
        # So we must keep Pareto-optimal set.
        
        # Algorithm for Pareto pruning (Maximize Length, Minimize Gap):
        # Sort by Length descending.
        # Iterate through sorted list.
        # Maintain a variable min_gap_encountered.
        # If current gap < min_gap_encountered, it is non-dominated.
        # Why? Because any state with higher length has been processed and had a gap >= min_gap_encountered (which is <= current gap).
        # Wait. If we have (5, 10) and (4, 5).
        # Sorted: (5, 10), (4, 5).
        # Process (5, 10): min_gap = inf. 10 < inf. Keep. min_gap = 10.
        # Process (4, 5): 5 < 10. Keep. min_gap = 5.
        # Result: [(5, 10), (4, 5)]. Correct.
        
        # If we have (5, 10) and (5, 8).
        # Sorted: (5, 8), (5, 10) (if we sort by gap asc for same length? Or just length desc).
        # If we sort by length desc, then for same length, we want the one with smallest gap.
        # Let's sort by (-length, gap).
        # (5, 8) comes before (5, 10).
        # Process (5, 8): min_gap = inf. 8 < inf. Keep. min_gap = 8.
        # Process (5, 10): 10 < 8? No. Discard.
        # Result: [(5, 8)]. Correct.
        
        if states[i]:
            # Sort by length descending, then gap ascending
            states[i].sort(key=lambda x: (-x[0], x[1]))
            pruned = []
            min_gap_so_far = float('inf')
            for l, g in states[i]:
                if g < min_gap_so_far:
                    pruned.append((l, g))
                    min_gap_so_far = g
            states[i] = pruned
            
            # Update global best if the top of this pruned list is better
            # The first element in pruned list has the max length.
            if pruned:
                max_l, min_g_for_max_l = pruned[0]
                if max_l > best_len:
                    best_len = max_l
                    best_idx = i
                    best_gap = min_g_for_max_l
                elif max_l == best_len and min_g_for_max_l < best_gap:
                    best_idx = i
                    best_gap = min_g_for_max_l

    # 3. Reconstruct the sequence
    # We have best_idx and best_gap.
    # We need to trace back.
    # We stored states, but we didn't store the predecessor index.
    # We need to modify the DP to store the predecessor.
    
    # Let's restart the DP with predecessor tracking.
    # states[i] will be a list of tuples (length, last_gap, prev_idx)
    
    states2 = [[] for _ in range(n_primes)]
    for i in range(n_primes):
        states2[i].append((1, 0, -1))
        
    best_len2 = 1
    best_idx2 = 0
    best_gap2 = 0
    
    for i in range(n_primes):
        p_i = primes[i]
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            for len_j, gap_j, prev_j in states2[j]:
                if gap_j <= gap:
                    new_len = len_j + 1
                    new_gap = gap
                    states2[i].append((new_len, new_gap, j))
                    
                    if new_len > best_len2:
                        best_len2 = new_len
                        best_idx2 = i
                        best_gap2 = new_gap
                    elif new_len == best_len2 and new_gap < best_gap2:
                        best_idx2 = i
                        best_gap2 = new_gap
                        
        # Prune states2[i]
        if states2[i]:
            states2[i].sort(key=lambda x: (-x[0], x[1]))
            pruned = []
            min_gap_so_far = float('inf')
            for l, g, p in states2[i]:
                if g < min_gap_so_far:
                    pruned.append((l, g, p))
                    min_gap_so_far = g
            states2[i] = pruned
            
            if pruned:
                max_l, min_g_for_max_l, _ = pruned[0]
                if max_l > best_len2:
                    best_len2 = max_l
                    best_idx2 = i
                    best_gap2 = min_g_for_max_l
                elif max_l == best_len2 and min_g_for_max_l < best_gap2:
                    best_idx2 = i
                    best_gap2 = min_g_for_max_l

    # Reconstruct
    seq_indices = []
    curr_idx = best_idx2
    curr_gap = best_gap2
    
    # Find the specific state in states2[curr_idx] that matches (best_len2, curr_gap)
    # There might be multiple states with same length and gap? No, pruned list has unique gaps for a given length?
    # Actually, the pruned list keeps the first one for a given gap value?
    # The sort is (-len, gap). If two states have same len and gap, the first one is kept, the second is discarded by the `g < min_gap_so_far` check?
    # If g == min_gap_so_far, it is discarded. So we keep the one with the smallest gap for a given length?
    # Wait, if we have (5, 10, j1) and (5, 10, j2).
    # Sort: (5, 10, j1), (5, 10, j2).
    # Process (5, 10, j1): 10 < inf. Keep. min_gap = 10.
    # Process (5, 10, j2): 10 < 10? False. Discard.
    # So we keep j1.
    
    target_state = None
    for l, g, p in states2[curr_idx]:
        if l == best_len2 and g == curr_gap:
            target_state = (l, g, p)
            break
            
    if target_state:
        _, _, prev = target_state
        seq_indices.append(curr_idx)
        while prev != -1:
            seq_indices.append(prev)
            # Find state in states2[prev] that leads to curr_idx
            # We need the state in states2[prev] that has length = current_len - 1 and gap = gap_prev
            # We don't have gap_prev directly, but we can infer it?
            # No, we need to find the specific state.
            # Let's just look for a state in states2[prev] that has prev_idx == curr_idx?
            # No, states2[prev] contains states ENDING at prev.
            # The state we want is the one that was extended to curr_idx.
            # We stored (new_len, new_gap, j) in states2[i].
            # So in states2[curr_idx], we found the state with prev=j.
            # Now we are at j. We need to find the state in states2[j] that was extended to curr_idx?
            # No, we need to trace back the chain.
            # The state in states2[curr_idx] is (L, G, j).
            # This means there was a state in states2[j] with (L-1, G_prev) such that G_prev <= G.
            # We need to find that state in states2[j].
            # Which one? The one that leads to the best solution.
            # Actually, we can just pick ANY state in states2[j] with length L-1?
            # No, we need the one that is part of the optimal path.
            # But since we are tracing back from the best, we can just look for a state in states2[j] with length L-1.
            # Is it unique? No.
            # But any state with length L-1 in states2[j] is a valid predecessor?
            # Not necessarily. It must be the one that allows the transition to curr_idx.
            # However, since we are reconstructing a valid sequence, we just need a valid path.
            # We know a path of length L exists ending at curr_idx.
            # We know a path of length L-1 exists ending at j.
            # We need to ensure the gap constraint holds.
            # Let's find the state in states2[j] with length L-1 and smallest gap?
            # Or just the first one?
            
            # Let's be precise. We will store the predecessor index in the state tuple?
            # We already have `prev` which is the index j.
            # Now we are at index j. We need to find the state in states2[j] that has length L-1.
            # There might be multiple. We should pick one that is "compatible" with the next step?
            # The next step was to curr_idx with gap G.
            # So we need a state in states2[j] with gap G_prev <= G.
            # Since the state (L-1, G_prev, k) was used to generate (L, G, j) in states2[curr_idx],
            # such a state must exist in states2[j] (before pruning? or after?).
            # Pruning might have removed it if it was dominated?
            # If it was dominated, there is another state (L'-1, G'_prev, k') with L'-1 >= L-1 and G'_prev <= G_prev.
            # If L'-1 > L-1, that's a longer sequence, which contradicts that L-1 is the max length ending at j?
            # No, L-1 is the length of the specific path.
            # If there was a longer path ending at j, we would have used that to make a longer path ending at curr_idx.
            # So L-1 MUST be the max length ending at j?
            # Not necessarily. We might have chosen a shorter path at j to get a smaller gap, allowing extension to curr_idx,
            # while a longer path at j had a larger gap and couldn't extend to curr_idx.
            # But if a longer path at j existed, it would have been a candidate for extension to curr_idx.
            # If it couldn't extend, it's not on the path.
            # So we need to find the specific state.
            
            # To avoid this complexity, let's just store the full path or use a backpointer array.
            # Given N is small, we can just store the full sequence in the state?
            # No, that's O(N^3) memory.
            
            # Let's just iterate through states2[j] and find one with length L-1 and gap <= G.
            # If there are multiple, pick the one with smallest gap?
            
            found_prev_state = False
            for l_p, g_p, k in states2[j]:
                if l_p == L - 1 and g_p <= G:
                    # This is a valid predecessor
                    # We need to continue tracing back from this state
                    # But we need to update L and G for the next iteration
                    # The gap of this state is g_p.
                    # The next gap (from j to curr_idx) is G.
                    # So the "current gap" for the next step back is g_p.
                    # And the "current length" is L-1.
                    # And the index is j.
                    # Wait, we are already at j.
                    # We need to find the predecessor of THIS state.
                    # The state is (L-1, g_p, k).
                    # So the next index to visit is k.
                    # And the gap to check against is g_p.
                    # And the length is L-1.
                    
                    # Let's set up for next iteration
                    L = L - 1
                    G = g_p
                    curr_idx = j
                    j = k # This is the new prev
                    # But we need to break the inner loop and continue outer?
                    # This logic is getting messy inside the while loop.
                    
                    # Let's restructure the reconstruction.
                    break
            if not found_prev_state:
                # This should not happen if DP is correct
                break
                
            # We need a cleaner way.
            # Let's just store the path in the state for small N?
            # N=1229. Max length ~20.
            # Storing a list of indices in each state is fine.
            
            # Let's restart DP with path storage.
            pass
            
    # Let's restart the DP with path storage to make reconstruction trivial.
    
    states3 = [[] for _ in range(n_primes)]
    for i in range(n_primes):
        states3[i].append((1, 0, [i]))
        
    best_len3 = 1
    best_path = [0]
    
    for i in range(n_primes):
        p_i = primes[i]
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            for len_j, gap_j, path_j in states3[j]:
                if gap_j <= gap:
                    new_len = len_j + 1
                    new_gap = gap
                    new_path = path_j + [i]
                    states3[i].append((new_len, new_gap, new_path))
                    
                    if new_len > best_len3:
                        best_len3 = new_len
                        best_path = new_path
                        
        # Prune states3[i]
        if states3[i]:
            states3[i].sort(key=lambda x: (-x[0], x[1]))
            pruned = []
            min_gap_so_far = float('inf')
            for l, g, p in states3[i]:
                if g < min_gap_so_far:
                    pruned.append((l, g, p))
                    min_gap_so_far = g
            states3[i] = pruned

    # Convert indices to primes
    result = [primes[idx] for idx in best_path]
    
    # Final validation
    if not result:
        return [2]
        
    # Check validity
    prev_gap = 0
    for k in range(1, len(result)):
        gap = result[k] - result[k-1]
        if gap < prev_gap:
            # Fallback
            return [2, 3]
        prev_gap = gap
        
    return result
# EVOLVE-BLOCK-END
