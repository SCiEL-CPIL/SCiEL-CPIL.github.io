import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 256
    
    # Precompute all APs: for each (d, s), store the list of indices
    # And for each position i, store which (d, s) APs contain it
    ap_indices = {}  # (d, s) -> list of indices
    pos_to_aps = [[] for _ in range(n)]  # i -> list of (d, s)
    
    for d in range(1, n):
        for s in range(n):
            idxs = []
            idx = s
            while idx < n:
                idxs.append(idx)
                idx += d
            if len(idxs) >= 1:
                key = (d, s)
                ap_indices[key] = idxs
                for i in idxs:
                    pos_to_aps[i].append(key)
    
    # Initial sequence: try a few random starts and pick best
    best_signs = None
    best_max = 10**9
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        rng = random.Random(42 + trial)
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        
        # Compute initial AP sums
        ap_sums = {}
        max_abs = 0
        for key, idxs in ap_indices.items():
            s = 0
            for i in idxs:
                s += signs[i]
            ap_sums[key] = s
            if abs(s) > max_abs:
                max_abs = abs(s)
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try flipping each position, pick the one that most reduces max
            best_gain = 0
            best_pos = -1
            for i in range(n):
                # Flipping i changes all APs containing i by ±2
                # Current sign is signs[i], flipping to -signs[i]
                # Delta = -2 * signs[i] for each AP containing i
                delta = -2 * signs[i]
                # Find new max after flipping
                # We need to check: for each AP containing i, new_sum = ap_sums[ap] + delta
                # The new max is max of:
                #   - APs NOT containing i: their sums unchanged
                #   - APs containing i: their sums changed by delta
                # We need to track the max efficiently
                # For now, compute new max by checking affected APs and keeping track
                # of the max of unaffected APs
                
                # To be efficient, precompute for each position the max abs sum
                # of APs NOT containing that position? That's expensive.
                
                # Alternative: just compute the new max by scanning all APs?
                # That's 65000 APs * 256 positions = too slow per iteration.
                
                # Better: maintain a data structure for AP sums.
                # For each flip, the new max = max(
                #   max over APs not containing i of |ap_sums[ap]|,
                #   max over APs containing i of |ap_sums[ap] + delta|
                # )
                
                # I'll maintain a sorted structure or just track the top few.
                # For simplicity in this first pass, let me just try a greedy
                # approach: for each position, compute what the new max would be
                # by only checking the affected APs and knowing the current max.
                
                # Actually, let me just do a full recompute of max for each candidate
                # flip, but only for the affected APs plus we need the max of
                # unaffected APs.
                
                # Let me precompute: for each position i, what is the max |sum|
                # over APs that do NOT contain i? This is expensive to maintain.
                
                # Alternative simpler approach: just try each flip and compute
                # the new global max by:
                # 1. For affected APs, compute new sums
                # 2. The new global max = max(max_affected_new, max_unaffected)
                # where max_unaffected is the max |sum| over APs not containing i.
                
                # To get max_unaffected efficiently, I can precompute for each i
                # the max |sum| over APs not containing i. But this changes as we
                # flip signs.
                
                # Let me use a different approach: maintain a count array for
                # |sum| values, or just recompute the max over all APs for each
                # candidate flip. With 65000 APs and 256 positions, that's
                # 16.6M operations per hill-climbing iteration, which might be
                # too slow.
                
                # Let me try: for each position i, the affected APs are those in
                # pos_to_aps[i]. The number is ~1500. For each, compute new sum.
                # For the unaffected, I need their max. I can precompute the top
                # few APs by |sum| and check if they're affected.
                
                # Actually, let me just do a simpler but still effective approach:
                # For each position i, compute the new max by:
                # - Taking the current max_abs
                # - For each affected AP, compute new |sum|
                # - The new max is max of (new affected max, and the max of
                #   unaffected APs)
                # To get max of unaffected, I can precompute for each i the
                # max |sum| over APs not in pos_to_aps[i]. But this is O(n * #APs)
                # to precompute and hard to update.
                
                # Let me just do the naive thing but optimize: for each candidate
                # flip, only recompute the max over affected APs and assume the
                # unaffected max is at most the current max_abs. This is an
                # overestimate but gives a valid upper bound. If the affected max
                # is less than current max_abs, the new max could still be
                # current max_abs (from unaffected APs). So I can't just use
                # affected max.
                
                # OK, let me just do a full recompute of the max for each
                # candidate. 65000 APs * 256 positions = 16.6M per iteration.
                # Each AP sum is just an integer, so this is ~16.6M integer
                # additions/comparisons. In Python, that might take ~1-2 seconds
                # per iteration. With 80 seconds, I can do ~40-80 iterations.
                # That's not great but let's try.
                
                # Actually, let me be smarter. I'll maintain the ap_sums dict
                # and for each candidate flip, I'll compute the new max by
                # iterating over all APs. But I can vectorize with numpy.
                
                pass  # Skip this inner loop for now, will redesign
        
        if best_signs is None or max_abs < best_max:
            best_max = max_abs
            best_signs = signs[:]
    
    return best_signs if best_signs else [1] * n
