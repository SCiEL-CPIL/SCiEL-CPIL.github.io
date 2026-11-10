import time
import numpy as np
import bisect

def construct():
    """
    Find the longest increasing subsequence of primes <= 10000
    with non-decreasing successive gaps.
    
    Uses Pareto-front DP: for each prime i, maintain Pareto-optimal states
    (last_gap, length) where no other state has both smaller gap and larger length.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # 1. Sieve primes up to 10000
    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0].tolist()
    n = len(primes)
    
    # For each prime index i, store Pareto front as parallel arrays:
    # front_gap[i]   : list of last_gap values (sorted ascending)
    # front_len[i]   : list of lengths (strictly increasing with gap)
    # front_prev[i]  : list of (prev_prime_idx, prev_gap) for reconstruction
    #
    # Invariant: front_gap[i][k] < front_gap[i][k+1] and
    #           front_len[i][k] < front_len[i][k+1]
    
    front_gap = [[] for _ in range(n)]
    front_len = [[] for _ in range(n)]
    front_prev = [[] for _ in range(n)]  # each entry: (prev_idx, prev_gap_val)
    
    # Initialize prime 0
    front_gap[0].append(-1)
    front_len[0].append(1)
    front_prev[0].append((-1, -1))
    
    best_len = 1
    best_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        pi = primes[i]
        
        # Collect all candidate (gap, length, prev_idx, prev_gap) tuples
        # Start with the "start new sequence" candidate
        cand_gap = [-1]
        cand_len = [1]
        cand_prev_idx = [-1]
        cand_prev_gap = [-1]
        
        # Try extending from each previous prime j
        for j in range(i):
            gap_ij = pi - primes[j]
            
            fj = front_gap[j]
            lj = front_len[j]
            pj = front_prev[j]
            
            # Use bisect to find the number of valid states in fj
            # We require gap_ij >= g_j.
            # fj is sorted ascending.
            # bisect_right(fj, gap_ij) gives the index where gap_ij would be inserted to maintain order.
            # All elements before this index are <= gap_ij.
            valid_count = bisect.bisect_right(fj, gap_ij)
            
            for k in range(valid_count):
                g_j = fj[k]
                l_j = lj[k]
                
                new_gap = gap_ij
                new_len = l_j + 1
                prev_idx = j
                prev_gap_val = g_j
                
                cand_gap.append(new_gap)
                cand_len.append(new_len)
                cand_prev_idx.append(prev_idx)
                cand_prev_gap.append(prev_gap_val)
        
        # Reduce to Pareto front:
        # Sort by (gap asc, length desc), then scan keeping only states
        # where length > max_length_seen_so_far
        
        m = len(cand_gap)
        if m <= 1:
            front_gap[i] = cand_gap[:]
            front_len[i] = cand_len[:]
            front_prev[i] = list(zip(cand_prev_idx, cand_prev_gap))
            if cand_len and cand_len[0] > best_len:
                best_len = cand_len[0]
                best_idx = i
            continue
        
        # Create list of tuples for efficient sorting
        # (gap, -length, prev_idx, prev_gap)
        # Sorting by gap asc, then -length asc (which is length desc)
        candidates = [(cand_gap[k], -cand_len[k], cand_prev_idx[k], cand_prev_gap[k]) for k in range(m)]
        candidates.sort()
        
        fg = []
        fl = []
        fp = []
        max_l = 0
        last_gap = -2  # sentinel, gaps are >= 0 or -1
        
        for g, neg_l, p_idx, p_gap in candidates:
            l = -neg_l
            # Deduplicate: skip if same gap and same or lower length
            if g == last_gap:
                if l <= max_l:
                    continue
                # Same gap but higher length: replace the last entry
                fg[-1] = g
                fl[-1] = l
                fp[-1] = (p_idx, p_gap)
                max_l = l
                continue
            if l > max_l:
                fg.append(g)
                fl.append(l)
                fp.append((p_idx, p_gap))
                max_l = l
                last_gap = g
        
        front_gap[i] = fg
        front_len[i] = fl
        front_prev[i] = fp
        
        if fl:
            if fl[-1] > best_len:
                best_len = fl[-1]
                best_idx = i
    
    # Reconstruct the sequence
    # Find the state in front at best_idx with length == best_len
    target_k = -1
    for k in range(len(front_len[best_idx])):
        if front_len[best_idx][k] == best_len:
            target_k = k
            break
    
    if target_k == -1:
        return [primes[0]]
    
    # Backtrack
    seq_indices = [best_idx]
    cur_idx = best_idx
    cur_k = target_k
    
    while True:
        prev_idx, prev_gap_val = front_prev[cur_idx][cur_k]
        if prev_idx == -1:
            break
        
        # Find the state in front_prev at prev_idx with gap == prev_gap_val
        # and length == front_len[cur_idx][cur_k] - 1
        target_len = front_len[cur_idx][cur_k] - 1
        found = False
        for k2 in range(len(front_gap[prev_idx])):
            if front_gap[prev_idx][k2] == prev_gap_val and front_len[prev_idx][k2] == target_len:
                seq_indices.append(prev_idx)
                cur_idx = prev_idx
                cur_k = k2
                found = True
                break
        
        if not found:
            # Should not happen; break to avoid infinite loop
            break
    
    seq_indices.reverse()
    result = [primes[i] for i in seq_indices]
    
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    if len(res) > 20:
        print(f"Sequence: {res[:10]}...{res[-10:]}")
    else:
        print(f"Sequence: {res}")
    # Verify
    if len(res) >= 2:
        gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
        is_non_decreasing = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
        print(f"Non-decreasing gaps: {is_non_decreasing}")
    print(f"All prime <= 10000: {all(p <= 10000 for p in res)}")
