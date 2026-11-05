"""Longest non-decreasing-gap prime subsequence via DAG longest path."""
# EVOLVE-BLOCK-START
import bisect

def construct():
    # Generate primes up to 10000 using sieve
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                for m in range(p*p, n + 1, p):
                    is_prime[m] = 0
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # For each node j, maintain a Pareto-optimal set of states.
    # A state is (last_gap, path_length, prev_node_idx, prev_gap).
    # Pareto-optimal means: no other state has both <= gap and >= length.
    # We store states sorted by gap (ascending), and lengths are strictly increasing
    # with gap (otherwise the state would be dominated).
    
    # states[j] = list of (gap, length, prev_idx, prev_gap) sorted by gap,
    #             with strictly increasing lengths.
    states = [None] * N
    
    # For backtracking, we also need to store the actual sequence info.
    # We'll use arrays: best_len[j] = max length ending at j
    #                  best_prev[j] = index of previous node in optimal path
    #                  best_gap[j] = the gap of the edge into j in optimal path
    
    best_len = [1] * N       # every node has at least length 1
    best_prev = [-1] * N     # -1 means start of sequence
    best_gap = [0] * N       # gap of incoming edge (0 for start)
    
    global_max_len = 1
    global_max_idx = 0
    
    for j in range(N):
        # states[j] will be a list of (gap, length, prev_idx, prev_gap)
        # Start with the trivial state: sequence of just [primes[j]]
        # Represented as gap=0, length=1, prev=-1, prev_gap=0
        temp = [(0, 1, -1, 0)]
        
        # Try extending sequences ending at i < j
        for i in range(j):
            si = states[i]
            if si is None:
                continue
            
            g_new = primes[j] - primes[i]
            
            # In si (sorted by gap, increasing lengths), find the state with
            # the largest gap <= g_new. That state has the maximum length
            # among all states with gap <= g_new.
            # Binary search: find rightmost index where gap <= g_new
            # si[k][0] is the gap
            gaps_i = [s[0] for s in si]
            pos = bisect.bisect_right(gaps_i, g_new) - 1
            if pos < 0:
                continue
            
            best_prev_state = si[pos]
            prev_gap_val = best_prev_state[0]
            prev_len = best_prev_state[1]
            prev_idx = best_prev_state[2]
            
            new_len = prev_len + 1
            
            # Add candidate state (g_new, new_len, i, prev_gap_val) to temp
            # We need to maintain Pareto optimality.
            # Since g_new is fixed for this (i,j) pair, and we want max length
            # for each gap, let's collect all candidates and then filter.
            temp.append((g_new, new_len, i, prev_gap_val))
        
        # Now filter temp to keep only Pareto-optimal states.
        # Group by gap, keep max length for each gap.
        best_by_gap = {}
        for (gap, length, pidx, pgap) in temp:
            if gap not in best_by_gap or length > best_by_gap[gap][1]:
                best_by_gap[gap] = (gap, length, pidx, pgap)
        
        # Sort by gap ascending
        unique_states = sorted(best_by_gap.values(), key=lambda x: x[0])
        
        # Apply Pareto filter: iterate by increasing gap, keep state only if
        # its length is strictly greater than all previous kept lengths.
        cleaned = []
        max_len_so_far = 0
        for state in unique_states:
            gap, length, pidx, pgap = state
            if length > max_len_so_far:
                cleaned.append(state)
                max_len_so_far = length
        
        states[j] = cleaned
        
        # Update global best
        if max_len_so_far > global_max_len:
            global_max_len = max_len_so_far
            global_max_idx = j
            # The best state is the last one in cleaned (highest length)
            best_state = cleaned[-1]
            best_len[j] = best_state[1]
            best_prev[j] = best_state[2]
            best_gap[j] = best_state[0]
        else:
            best_len[j] = max_len_so_far
    
    # Backtrack from global_max_idx
    seq = []
    curr = global_max_idx
    while curr != -1:
        seq.append(primes[curr])
        curr = best_prev[curr]
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
