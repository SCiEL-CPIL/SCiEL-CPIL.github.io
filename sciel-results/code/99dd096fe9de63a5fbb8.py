import time
import bisect
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes[:]
    
    # dp[u][v] length of longest seq ending with primes[u], primes[v]
    dp = np.zeros((n, n), dtype=np.uint8)
    parent = np.full((n, n), -1, dtype=np.int16)
    
    structures = [None] * n
    
    best_len = 2
    best_u = 0
    best_v = 1
    
    # Precompute primes as numpy array for faster subtraction?
    # primes_np = np.array(primes, dtype=np.int32)
    
    for v in range(1, n):
        pv = primes[v]
        
        for u in range(0, v):
            pu = primes[u]
            gap_v_u = pv - pu
            
            if structures[u] is not None:
                keys, prefix_maxes, prefix_argmaxes = structures[u]
                pos = bisect.bisect_right(keys, gap_v_u)
                if pos > 0:
                    idx = pos - 1
                    max_prev_len = prefix_maxes[idx]
                    best_w = prefix_argmaxes[idx]
                    curr_len = max_prev_len + 1
                    dp[u, v] = curr_len
                    parent[u, v] = best_w
                else:
                    curr_len = 2
                    dp[u, v] = curr_len
                    parent[u, v] = -1
            else:
                curr_len = 2
                dp[u, v] = curr_len
                parent[u, v] = -1
                
            if curr_len > best_len:
                best_len = curr_len
                best_u = u
                best_v = v
        
        # Build structure for index v
        if v > 0:
            # keys_v = primes[v] - primes[0..v-1]
            # This is an increasing sequence.
            # vals_v = dp[0..v-1, v]
            
            # Using numpy for speed
            # primes is a list, so primes[:v] creates a list.
            # Converting to numpy array each time is O(v).
            # Total O(N^2).
            
            # We can keep primes as a numpy array.
            # But primes is a list of ints.
            # Let's convert once.
            
            pass # Handled outside? No, inside loop.
            
            # Let's optimize the structure building.
            # We need to sort by key.
            # keys are p_v - p_w.
            # Since p_w is increasing, keys are decreasing as w increases?
            # w=0 -> p_v - p_0 (largest)
            # w=v-1 -> p_v - p_{v-1} (smallest)
            # So keys are decreasing with w.
            # We want increasing keys for bisect.
            # So we reverse the order of w.
            # w goes v-1, v-2, ..., 0.
            # keys: p_v - p_{v-1}, p_v - p_{v-2}, ..., p_v - p_0.
            # This is increasing.
            
            # vals: dp[v-1, v], dp[v-2, v], ..., dp[0, v]
            
            # We can just iterate w from v-1 down to 0 and build the lists.
            # This avoids numpy overhead for small v?
            # Or use numpy for large v?
            
            keys_list = []
            vals_list = []
            ws_list = []
            
            # We need to append in order of increasing key.
            # So w from v-1 down to 0.
            for w in range(v-1, -1, -1):
                key = pv - primes[w]
                val = dp[w, v]
                keys_list.append(key)
                vals_list.append(val)
                ws_list.append(w)
            
            # Now compute prefix maxes
            prefix_maxes = [0] * v
            prefix_argmaxes = [0] * v
            
            prefix_maxes[0] = vals_list[0]
            prefix_argmaxes[0] = ws_list[0]
            
            for i in range(1, v):
                if vals_list[i] > prefix_maxes[i-1]:
                    prefix_maxes[i] = vals_list[i]
                    prefix_argmaxes[i] = ws_list[i]
                else:
                    prefix_maxes[i] = prefix_maxes[i-1]
                    prefix_argmaxes[i] = prefix_argmaxes[i-1]
                    
            structures[v] = (keys_list, prefix_maxes, prefix_argmaxes)
        else:
            structures[v] = None
            
        if time.time() > deadline:
            break

    # Reconstruct sequence
    seq = [primes[best_v], primes[best_u]]
    cur_u = best_u
    cur_v = best_v
    
    while parent[cur_u, cur_v] != -1:
        w = parent[cur_u, cur_v]
        seq.append(primes[w])
        cur_v = cur_u
        cur_u = w
        
    seq.reverse()
    
    return seq
