import time
import numpy as np

def construct():
    """
    Find the longest increasing subsequence of primes <= 10000 
    with non-decreasing successive gaps.
    
    Strategy:
    1. Sieve primes up to 10000.
    2. Dynamic Programming with Pareto fronts:
       For each prime i, maintain a set of Pareto-optimal states (last_gap, length).
       A state (g, l) is dominated if there exists (g', l') such that g' <= g and l' >= l.
       Transitions:
       - Start new sequence at i: state (-1, 1)
       - Extend sequence ending at j to i:
         If state at j is (g_j, l_j):
           If g_j == -1: new state at i is (gap_ij, l_j + 1)
           Else if gap_ij >= g_j: new state at i is (gap_ij, l_j + 1)
    3. Reconstruct the path using stored predecessors.
    4. Use NumPy for vectorized gap calculations and filtering to stay within time limits.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Sieve primes up to 10000
    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0].tolist()
    n = len(primes)
    
    # Pareto fronts for each prime index
    # pareto[i] is a list of (last_gap, length, prev_index, prev_gap) tuples
    # We store prev_index and prev_gap to allow reconstruction without complex backtracking logic
    pareto = [[] for _ in range(n)]
    
    # Initialize the first prime
    # State: (last_gap, length, prev_idx, prev_gap_in_prev_state)
    # For the first prime, last_gap is -1 (sentinel), length 1, no prev
    pareto[0].append((-1, 1, -1, -1))
    
    best_len = 1
    best_idx = 0
    
    for i in range(1, n):
        pi = primes[i]
        
        # Candidates for pareto[i]
        # Always can start a new sequence here
        candidates = [(-1, 1, -1, -1)]
        
        # Check all previous primes j < i
        # Vectorized calculation of gaps
        # We need to iterate through pareto[j] for each j, which is hard to fully vectorize 
        # across j because pareto[j] has variable length. 
        # However, n is small (1229). The inner loop is over pareto[j] which is small.
        # We can optimize by precomputing gaps for all j < i? 
        # gaps[i, j] = pi - primes[j]
        
        # Let's just loop j, but keep it tight.
        for j in range(i):
            if time.time() > deadline:
                break
                
            pj = primes[j]
            gap_ij = pi - pj
            
            # Iterate through valid states ending at j
            for state in pareto[j]:
                g_j, l_j, prev_idx_j, prev_gap_j = state
                
                # Check if we can extend
                if g_j == -1:
                    # No prior gap constraint, any gap works
                    new_gap = gap_ij
                    new_len = l_j + 1
                    # Predecessor is j with state (g_j, l_j, ...)
                    # We need to record which state at j we came from to reconstruct
                    # Since pareto[j] is a list, we can store the index in the list? 
                    # Or just store the state values. Storing values is safer.
                    candidates.append((new_gap, new_len, j, g_j))
                else:
                    if gap_ij >= g_j:
                        new_gap = gap_ij
                        new_len = l_j + 1
                        candidates.append((new_gap, new_len, j, g_j))
        
        # Reduce candidates to Pareto front
        # Sort by gap ascending, then length descending
        # We want to keep states where no other state has (gap' <= gap AND len' >= len)
        # Sorting by gap asc: 
        #   If we iterate, we keep a state if its length is strictly greater than the max length seen so far for smaller/equal gaps.
        
        # Optimization: Remove duplicates? (gap, len) pairs might repeat.
        # Using a set for (gap, len) might help, but we need to keep the predecessor info.
        # Actually, if two candidates have the same (gap, len), we only need one.
        # Let's deduplicate based on (gap, len) keeping the first one found (arbitrary predecessor is fine).
        
        unique_candidates = {}
        for c in candidates:
            key = (c[0], c[1])
            if key not in unique_candidates:
                unique_candidates[key] = c
        
        cand_list = list(unique_candidates.values())
        
        if not cand_list:
            # Should not happen, at least the start state exists
            continue
            
        cand_list.sort(key=lambda x: (x[0], -x[1]))
        
        front = []
        max_len_seen = 0
        for c in cand_list:
            gap, length, prev_idx, prev_gap = c
            if length > max_len_seen:
                front.append(c)
                max_len_seen = length
        
        pareto[i] = front
        
        # Update global best
        if front:
            # The best length in this front is the last element (since sorted by gap asc, length is increasing in the front)
            # Wait, is length strictly increasing in the Pareto front?
            # Yes, because we only add if length > max_len_seen.
            best_in_front = front[-1]
            if best_in_front[1] > best_len:
                best_len = best_in_front[1]
                best_idx = i
    
    # Reconstruct the sequence
    # We need to find the state in pareto[best_idx] with length == best_len
    # There might be multiple states with the same max length? 
    # In a Pareto front, for a given length, only the one with the smallest gap is kept?
    # No, we keep (gap, len) if len > max_len_seen. 
    # So for a specific max_len, there is only one state in the front with that length?
    # Yes, because if there were two states with same length, the one with smaller gap would dominate the other (same len, smaller gap).
    # So the state with max length is unique in the front.
    
    target_state = None
    for state in pareto[best_idx]:
        if state[1] == best_len:
            target_state = state
            break
    
    if target_state is None:
        # Fallback
        return [primes[0]]
        
    seq_indices = [best_idx]
    cur_idx = best_idx
    cur_state = target_state
    
    # Backtrack
    # cur_state is (gap, len, prev_idx, prev_gap)
    # We need to find the state in pareto[prev_idx] that matches (prev_gap, len-1)
    # Note: The 'prev_gap' stored in cur_state is the 'last_gap' of the predecessor state.
    # So we look for a state in pareto[prev_idx] with last_gap == prev_gap and length == cur_state[1] - 1.
    
    while cur_state[2] != -1:
        prev_idx = cur_state[2]
        prev_gap_val = cur_state[3]
        prev_len = cur_state[1] - 1
        
        # Find the matching state in pareto[prev_idx]
        prev_state = None
        for s in pareto[prev_idx]:
            if s[0] == prev_gap_val and s[1] == prev_len:
                prev_state = s
                break
        
        if prev_state is None:
            # Error in logic? Should not happen if we stored correctly.
            # Fallback: break
            break
            
        seq_indices.append(prev_idx)
        cur_idx = prev_idx
        cur_state = prev_state
        
    seq_indices.reverse()
    result = [primes[i] for i in seq_indices]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res[:10]}...{res[-10:]}")
    # Verify
    gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
    is_non_decreasing = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
    print(f"Non-decreasing gaps: {is_non_decreasing}")
    print(f"All prime <= 10000: {all(p <= 10000 for p in res)}")
