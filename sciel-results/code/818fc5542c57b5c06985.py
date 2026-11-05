"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    import bisect
    
    # For each prime index i:
    #   dp_gaps[i] = sorted list of gap values
    #   dp_prefix[i] = list of prefix max lengths corresponding to dp_gaps[i]
    #   dp_raw[i] = dict mapping gap -> max_length
    #   pred[i][gap] = (prev_index, prev_gap) for backtracking
    
    dp_gaps = [None] * n
    dp_prefix = [None] * n
    dp_raw = [None] * n
    pred = [None] * n
    
    best_len = 1
    best_j = 0
    best_end_gap = 0
    
    for j in range(n):
        if time.time() > deadline:
            break
        
        candidates = {0: 1}
        preds = {0: (None, None)}  # gap -> (prev_idx, prev_gap)
        
        for i in range(j):
            gap = primes[j] - primes[i]
            sg = dp_gaps[i]
            if sg is None:
                continue
            pos = bisect.bisect_right(sg, gap)
            if pos > 0:
                best_prev = dp_prefix[i][pos - 1]
                if best_prev > 0:
                    new_len = best_prev + 1
                    if gap not in candidates or candidates[gap] < new_len:
                        candidates[gap] = new_len
                        # Find which prev_gap in dp_raw[i] achieves best_prev with gap' <= gap
                        # We need to find a gap' <= gap such that dp_raw[i][gap'] == best_prev
                        # Use the prefix structure: find the rightmost gap' <= gap where prefix max first reaches best_prev
                        # Simpler: iterate through dp_raw[i] items
                        prev_gap_found = None
                        for g2, l2 in dp_raw[i].items():
                            if g2 <= gap and l2 == best_prev:
                                prev_gap_found = g2
                                break
                        if prev_gap_found is None:
                            # Fallback: find any g2 <= gap with l2 >= best_prev - 1 (shouldn't happen)
                            for g2, l2 in dp_raw[i].items():
                                if g2 <= gap and l2 >= best_prev:
                                    prev_gap_found = g2
                                    break
                        if prev_gap_found is not None:
                            preds[gap] = (i, prev_gap_found)
        
        if not candidates:
            candidates = {0: 1}
            preds = {0: (None, None)}
        
        dp_raw[j] = candidates
        pred[j] = preds
        
        # Build sorted gap list and prefix max
        items = sorted(candidates.items())
        gaps_list = [x[0] for x in items]
        prefix = []
        cur_max = 0
        for gap, length in items:
            cur_max = max(cur_max, length)
            prefix.append(cur_max)
        
        dp_gaps[j] = gaps_list
        dp_prefix[j] = prefix
        
        max_len_j = max(candidates.values())
        if max_len_j > best_len:
            best_len = max_len_j
            best_j = j
            # Find the gap that achieves this max
            for gap, length in candidates.items():
                if length == max_len_j:
                    best_end_gap = gap
                    break
    
    # Backtrack to reconstruct the sequence
    if best_j >= 0 and best_len > 1:
        seq = [primes[best_j]]
        cur_j = best_j
        cur_gap = best_end_gap
        cur_len = best_len
        
        while cur_len > 1:
            if time.time() > deadline:
                break
            
            p = pred[cur_j]
            if p is None or cur_gap not in p:
                break
            
            prev_idx, prev_gap = p[cur_gap]
            if prev_idx is None:
                break
            
            seq.append(primes[prev_idx])
            cur_j = prev_idx
            cur_gap = prev_gap
            cur_len -= 1
        
        seq.reverse()
        
        # Verify
        if len(seq) >= 2:
            gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            if all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
                return seq
        
    # Fallback: greedy
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    
    return values
# EVOLVE-BLOCK-END
