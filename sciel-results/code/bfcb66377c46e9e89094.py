import sys
import time
import bisect

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin

    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    # dp[j] is a list of states: (gap, length, parent_prime_index, parent_state_index)
    # sorted by gap.
    dp = [[] for _ in range(N)]
    
    # prefix_max_len[j][k] = max length in dp[j][0..k]
    # prefix_argmax[j][k] = index in dp[j] of the state achieving prefix_max_len[j][k]
    prefix_max_len = [[] for _ in range(N)]
    prefix_argmax = [[] for _ in range(N)]
    
    best_len = 1
    best_end = 0
    
    # Initialize j=0
    dp[0] = [(0, 1, -1, -1)]
    prefix_max_len[0] = [1]
    prefix_argmax[0] = [0]
    
    for j in range(1, N):
        p_j = primes[j]
        candidates = [(0, 1, -1, -1)] # Start with self
        
        for i in range(j):
            p_i = primes[i]
            g = p_j - p_i
            
            # Query dp[i] for max length with gap <= g
            if not dp[i]:
                continue
                
            # Binary search for the largest gap <= g in dp[i]
            # We need a list of gaps for dp[i]
            # To avoid creating this list every time, we can store gaps separately or just use bisect on a list of gaps.
            # Since N is small, creating the list is fine, or we can just iterate if N is very small.
            # But binary search is better.
            
            # Let's extract gaps. This is O(N) per i, so O(N^2) total. Acceptable.
            gaps_i = [s[0] for s in dp[i]]
            
            # Find rightmost index where gap <= g
            # bisect_right gives the index to insert g to keep sorted order.
            # We want the last element <= g.
            # If all gaps are < g, bisect_right returns len(gaps). We want len-1.
            # If some gaps are > g, bisect_right returns the index of the first > g. We want index-1.
            
            idx = bisect.bisect_right(gaps_i, g)
            if idx == 0:
                continue # No gap <= g (except maybe 0, but 0 is always there)
                
            res_idx = idx - 1
            
            # Get the best length up to res_idx
            best_l = prefix_max_len[i][res_idx]
            argmax_idx = prefix_argmax[i][res_idx]
            
            new_len = best_l + 1
            # State: (gap, length, parent_prime_index, parent_state_index)
            candidates.append((g, new_len, i, argmax_idx))
            
        # Process candidates for j
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        dp[j] = candidates
        
        # Compute prefix_max_len[j] and prefix_argmax[j]
        curr_max = 0
        curr_argmax = 0
        for k in range(len(candidates)):
            if candidates[k][1] > curr_max:
                curr_max = candidates[k][1]
                curr_argmax = k
            prefix_max_len[j].append(curr_max)
            prefix_argmax[j].append(curr_argmax)
            
        if curr_max > best_len:
            best_len = curr_max
            best_end = j
            
        # Time check
        if time.time() > DEADLINE:
            break

    # Reconstruct path
    if best_len <= 1:
        return [primes[0]]
        
    path = []
    curr_j = best_end
    # Find the state in dp[curr_j] that has length best_len
    # It is at prefix_argmax[curr_j][-1]
    state_idx = prefix_argmax[curr_j][-1]
    
    while curr_j != -1:
        path.append(primes[curr_j])
        state = dp[curr_j][state_idx]
        parent_j = state[2]
        parent_state_idx = state[3]
        if parent_j == -1:
            break
        curr_j = parent_j
        state_idx = parent_state_idx
        
    path.reverse()
    return path

if __name__ == "__main__":
    construct()
