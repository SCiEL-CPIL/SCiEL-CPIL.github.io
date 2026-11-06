import random
import time

N = 128
APS = []
POS_APS = [[] for _ in range(N)]
NUM_APS = 0

def _init_data():
    global APS, POS_APS, NUM_APS
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = NUM_APS
            APS.append((a, b, c, e))
            POS_APS[a].append(idx)
            POS_APS[b].append(idx)
            POS_APS[c].append(idx)
            POS_APS[e].append(idx)
            NUM_APS += 1

_init_data()

def _local_search_fast(k, seed, time_limit):
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize colors randomly
    colors = [rng.randrange(k) for _ in range(N)]
    
    # Precompute distinct counts for each AP
    ap_distinct = [0] * NUM_APS
    # ap_hist is a list of lists, ap_hist[i][color] = count
    ap_hist = [[0] * k for _ in range(NUM_APS)]
    
    for idx, (a, b, c, e) in enumerate(APS):
        h = ap_hist[idx]
        ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
        h[ca] += 1
        h[cb] += 1
        h[cc] += 1
        h[ce] += 1
        distinct = 0
        for count in h:
            if count > 0:
                distinct += 1
        ap_distinct[idx] = distinct
        
    total_bad = sum(1 for d in ap_distinct if d < 3)
    
    if total_bad == 0:
        return colors

    best_colors = colors[:]
    best_bad = total_bad
    bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
    step = 0

    while time.time() - start_time < time_limit:
        if total_bad == 0:
            return colors
            
        if not bad_aps:
            bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
            total_bad = len(bad_aps)
            if not bad_aps:
                return colors

        # Pick a random bad AP
        ap_idx = bad_aps[rng.randrange(len(bad_aps))]
        a, b, c, e = APS[ap_idx]
        
        # Pick a random position in this AP to change
        pos_idx = rng.randrange(4)
        if pos_idx == 0:
            i = a
        elif pos_idx == 1:
            i = b
        elif pos_idx == 2:
            i = c
        else:
            i = e
            
        old_color = colors[i]
        new_color = rng.randrange(k)
        
        if new_color == old_color:
            new_color = (new_color + 1) % k
            
        delta = 0
        # Evaluate change
        for ap_i in POS_APS[i]:
            h = ap_hist[ap_i]
            D = ap_distinct[ap_i]
            
            count_old = h[old_color]
            count_new = h[new_color]
            
            # Calculate new distinct count
            new_D = D
            if count_old == 1:
                new_D -= 1
            if count_new == 0:
                new_D += 1
                
            old_bad = 1 if D < 3 else 0
            new_bad = 1 if new_D < 3 else 0
            delta += new_bad - old_bad
            
        # Acceptance criteria
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.3
        else:
            accept = rng.random() < 0.05
            
        if not accept:
            continue
            
        # Apply change
        colors[i] = new_color
        total_bad += delta
        
        for ap_i in POS_APS[i]:
            h = ap_hist[ap_i]
            D = ap_distinct[ap_i]
            
            h[old_color] -= 1
            if h[old_color] == 0:
                D -= 1
                
            h[new_color] += 1
            if h[new_color] == 1:
                D += 1
                
            ap_distinct[ap_i] = D
            
        step += 1
        if total_bad < best_bad:
            best_bad = total_bad
            best_colors = colors[:]
            
        # Rebuild bad_aps periodically to ensure we don't miss new bad APs created by changes
        # that were previously good, or to clean up the list if it grows stale.
        if step % 500 == 0:
            bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
            total_bad = len(bad_aps)
            if not bad_aps:
                return colors

    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_colors = None
    best_num_colors = N + 1
    
    # Try small k first
    for k in [3, 4, 5]:
        if time.time() > deadline - 5.0:
            break
            
        # Use most of the remaining time for this k
        time_slice = deadline - time.time() - 2.0
        if time_slice < 1.0:
            break
            
        seed = int(time.time() * 1000) % (2**32)
        
        res = _local_search_fast(k, seed, time_slice)
        
        # Validate
        valid = True
        for (a, b, c, e) in APS:
            ca, cb, cc, ce = res[a], res[b], res[c], res[e]
            if len({ca, cb, cc, ce}) < 3:
                valid = False
                break
        
        if valid:
            num_used = len(set(res))
            if num_used < best_num_colors:
                best_num_colors = num_used
                best_colors = res[:]
            if k == 3:
                return res

    # If no small k worked, try a larger k with remaining time
    if best_colors is None and time.time() < deadline - 1.0:
        time_slice = deadline - time.time() - 1.0
        if time_slice > 0:
            res = _local_search_fast(10, 42, time_slice)
            valid = True
            for (a, b, c, e) in APS:
                ca, cb, cc, ce = res[a], res[b], res[c], res[e]
                if len({ca, cb, cc, ce}) < 3:
                    valid = False
                    break
            if valid:
                best_colors = res

    if best_colors is None:
        return [i % 3 for i in range(N)]

    return best_colors
