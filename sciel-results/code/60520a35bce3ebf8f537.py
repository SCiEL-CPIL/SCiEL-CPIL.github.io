import time
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0].tolist()
    n = len(primes)
    
    # front_gap[i]: list of gaps (sorted asc)
    # front_len[i]: list of lengths (sorted asc, corresponding to gaps)
    # front_prev[i]: list of (prev_idx, prev_k) for reconstruction
    # Note: For length 1, we can store prev_idx = -1, prev_k = -1.
    # The "gap" for length 1 is 0.
    
    front_gap = [[] for _ in range(n)]
    front_len = [[] for _ in range(n)]
    front_prev = [[] for _ in range(n)]
    
    # Initialize i=0
    front_gap[0] = [0]
    front_len[0] = [1]
    front_prev[0] = [(-1, -1)]
    
    best_len = 1
    best_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        
        # Candidates: list of (gap, length, prev_idx, prev_k)
        # Start new sequence
        candidates = [(0, 1, -1, -1)]
        
        for j in range(i):
            gap_ij = pi - primes[j]
            
            fj = front_gap[j]
            lj = front_len[j]
            pj = front_prev[j]
            
            # Find max k such that fj[k] <= gap_ij
            # fj is sorted ascending
            k = bisect.bisect_right(fj, gap_ij) - 1
            
            if k >= 0:
                # The best state to extend is at index k
                # because lj is increasing with k
                new_len = lj[k] + 1
                # new_gap is gap_ij
                candidates.append((gap_ij, new_len, j, k))
        
        # Process candidates to form Pareto front for i
        # Sort by gap asc, then length desc (to handle duplicates)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        fg = []
        fl = []
        fp = []
        max_l = 0
        
        # We need to deduplicate gaps and keep max length, then filter by increasing length
        # Since sorted by gap, we can iterate
        # But we also need to ensure length is increasing for the Pareto front
        
        # Let's first reduce to max length per gap
        # Then filter
        # Actually, the sort (gap asc, len desc) allows us to just take the first one for each gap
        # Then check if len > max_l
        
        last_gap = None
        for g, l, p_idx, p_k in candidates:
            if g == last_gap:
                # Since sorted by len desc, the first one has max len for this gap
                # Subsequent ones have lower len, so skip
                continue
            
            if l > max_l:
                fg.append(g)
                fl.append(l)
                fp.append((p_idx, p_k))
                max_l = l
                last_gap = g
            else:
                # If l <= max_l, this point is dominated by a previous point with smaller gap and larger/equal length
                # So we don't add it. But we still update last_gap?
                # No, if l <= max_l, it's not in the front.
                # However, for the "last_gap" logic:
                # We skip if g == last_gap.
                # If g != last_gap, we check if it's in front.
                # If it's not in front, we don't add it.
                # Do we update last_gap?
                # last_gap is used to skip duplicates.
                # If we skip adding it, we should still update last_gap to g?
                # Yes, because subsequent entries with same g should be skipped.
                last_gap = g

        front_gap[i] = fg
        front_len[i] = fl
        front_prev[i] = fp
        
        if fl:
            if fl[-1] > best_len:
                best_len = fl[-1]
                best_idx = i
    
    # Reconstruct
    if best_len == 1:
        return [primes[best_idx]]
        
    # Find the state in front_len[best_idx] with length == best_len
    target_k = -1
    for k in range(len(front_len[best_idx])):
        if front_len[best_idx][k] == best_len:
            target_k = k
            break
            
    if target_k == -1:
        return [primes[best_idx]]
        
    seq_indices = [best_idx]
    cur_idx = best_idx
    cur_k = target_k
    
    while True:
        prev_idx, prev_k = front_prev[cur_idx][cur_k]
        if prev_idx == -1:
            break
        seq_indices.append(prev_idx)
        cur_idx = prev_idx
        cur_k = prev_k
        
    seq_indices.reverse()
    result = [primes[i] for i in seq_indices]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    if len(res) >= 2:
        gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
        is_non_decreasing = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
        print(f"Non-decreasing gaps: {is_non_decreasing}")
