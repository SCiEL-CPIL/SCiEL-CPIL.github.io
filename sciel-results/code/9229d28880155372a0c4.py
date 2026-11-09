import time
import random
import numpy as np

def count_mono_4ap(colors):
    n = 256
    c = colors
    count = 0
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i = a - 1
            col = c[i]
            if col == c[i + d] == c[i + 2 * d] == c[i + 3 * d]:
                count += 1
    return count

def build_affected(n=256):
    """
    For each position p (0-indexed), store arrays of indices of the 4-APs containing p.
    Returns a list of 4 arrays per position: idx0, idx1, idx2, idx3.
    """
    affected = []
    for p in range(n):
        idxs0 = []
        idxs1 = []
        idxs2 = []
        idxs3 = []
        # d is the step size
        for d in range(1, 86):
            # a is the start of the AP (1-indexed), so a-1 is 0-indexed start
            # The AP is a-1, a-1+d, a-1+2d, a-1+3d
            # p must be one of these.
            # p = a-1 + k*d for k in {0,1,2,3}
            # a-1 = p - k*d
            # a must be in [1, n-3d+1] => a-1 in [0, n-3d]
            # So p - k*d must be in [0, n-3d]
            # And p must be >= k*d (since a-1 >= 0)
            # Also a-1+3d = p - k*d + 3d = p + (3-k)d must be < n
            for k in range(4):
                start = p - k * d
                end = start + 3 * d
                if 0 <= start and end < n:
                    # This AP is valid. Record its indices.
                    idxs0.append(start)
                    idxs1.append(start + d)
                    idxs2.append(start + 2 * d)
                    idxs3.append(start + 3 * d)
        
        affected.append((
            np.array(idxs0, dtype=np.int32) if idxs0 else np.array([], dtype=np.int32),
            np.array(idxs1, dtype=np.int32) if idxs1 else np.array([], dtype=np.int32),
            np.array(idxs2, dtype=np.int32) if idxs2 else np.array([], dtype=np.int32),
            np.array(idxs3, dtype=np.int32) if idxs3 else np.array([], dtype=np.int32)
        ))
    return affected

def thue_morse(n):
    seq = [0] * n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def local_search(colors, deadline, affected):
    c = np.array(colors, dtype=np.int8)
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            old = c[p]
            c[p] = 1 - old
            # Compute delta using vectorized operations
            idx0, idx1, idx2, idx3 = affected[p]
            if len(idx0) == 0:
                continue
            
            # Get colors of the other 3 elements in each AP
            # We need to check if the AP was mono before and after.
            # Before: c[p] was `old`.
            # After: c[p] is `1-old`.
            
            # Create arrays of colors for the 4 positions for all APs involving p.
            # pos0
            col0_before = np.where(idx0 == p, old, c[idx0])
            col0_after = np.where(idx0 == p, 1 - old, c[idx0])
            
            # pos1
            col1_before = np.where(idx1 == p, old, c[idx1])
            col1_after = np.where(idx1 == p, 1 - old, c[idx1])
            
            # pos2
            col2_before = np.where(idx2 == p, old, c[idx2])
            col2_after = np.where(idx2 == p, 1 - old, c[idx2])
            
            # pos3
            col3_before = np.where(idx3 == p, old, c[idx3])
            col3_after = np.where(idx3 == p, 1 - old, c[idx3])
            
            # Check mono
            mono_before = (col0_before == col1_before) & (col1_before == col2_before) & (col2_before == col3_before)
            mono_after = (col0_after == col1_after) & (col1_after == col2_after) & (col2_after == col3_after)
            
            delta = np.sum(mono_after) - np.sum(mono_before)
            
            new = cur + delta
            if new < cur:
                cur = new
                improved = True
            else:
                c[p] = old
    return c, cur

def simulated_annealing(colors, deadline, affected, start_temp=10.0, cool_rate=0.9995, min_temp=0.01):
    c = np.array(colors, dtype=np.int8)
    cur = count_mono_4ap(c)
    best = c.copy()
    best_count = cur
    temp = start_temp
    
    while temp > min_temp and time.time() < deadline:
        p = random.randint(0, 255)
        old = c[p]
        c[p] = 1 - old
        
        # Compute delta using vectorized operations
        idx0, idx1, idx2, idx3 = affected[p]
        if len(idx0) == 0:
            delta = 0
        else:
            col0_before = np.where(idx0 == p, old, c[idx0])
            col0_after = np.where(idx0 == p, 1 - old, c[idx0])
            col1_before = np.where(idx1 == p, old, c[idx1])
            col1_after = np.where(idx1 == p, 1 - old, c[idx1])
            col2_before = np.where(idx2 == p, old, c[idx2])
            col2_after = np.where(idx2 == p, 1 - old, c[idx2])
            col3_before = np.where(idx3 == p, old, c[idx3])
            col3_after = np.where(idx3 == p, 1 - old, c[idx3])
            
            mono_before = (col0_before == col1_before) & (col1_before == col2_before) & (col2_before == col3_before)
            mono_after = (col0_after == col1_after) & (col1_after == col2_after) & (col2_after == col3_after)
            
            delta = int(np.sum(mono_after) - np.sum(mono_before))
        
        new = cur + delta
        if new < cur:
            cur = new
            if new < best_count:
                best_count = new
                best = c.copy()
        else:
            # Accept with probability exp(-delta/temp)
            if random.random() < np.exp(-delta / temp):
                cur = new
            else:
                c[p] = old
        
        temp *= cool_rate
    
    return best, best_count

def construct():
    start = time.time()
    deadline = start + 110  # Safer margin
    
    affected = build_affected(256)
    
    candidates = []
    candidates.append(thue_morse(256))
    candidates.append([(i // 2) % 2 for i in range(256)])
    candidates.append([(i // 3) % 2 for i in range(256)])
    candidates.append([(i // 4) % 2 for i in range(256)])
    tm_full = thue_morse(257)
    candidates.append(tm_full[1:257])
    
    def v2_color(n):
        out = []
        for i in range(1, n + 1):
            x = i
            p = 0
            while x % 2 == 0:
                x //= 2
                p += 1
            out.append(p % 2)
        return out
    candidates.append(v2_color(256))
    candidates.append([(i % 2) ^ ((i // 2) % 2) for i in range(256)])
    
    # Add some random candidates
    for _ in range(3):
        candidates.append([random.randint(0, 1) for _ in range(256)])
    
    best = None
    best_count = float('inf')
    
    # First pass: evaluate candidates and run greedy local search on each
    for cand in candidates:
        if time.time() > deadline - 40:  # Leave 40s for SA and final polish
            break
        cand_arr = np.array(cand, dtype=np.int8)
        cnt = count_mono_4ap(cand_arr)
        if cnt < best_count:
            best_count = cnt
            best = cand_arr.copy()
        # Run local search on this candidate
        ls_result, ls_count = local_search(cand_arr, deadline - 40, affected)
        if ls_count < best_count:
            best_count = ls_count
            best = ls_result
    
    if best is None:
        best = np.array(candidates[0], dtype=np.int8)
        best_count = count_mono_4ap(best)
    
    # Run simulated annealing from the best result
    if time.time() < deadline - 10:
        sa_result, sa_count = simulated_annealing(best, deadline - 10, affected, start_temp=8.0, cool_rate=0.9997, min_temp=0.01)
        if sa_count < best_count:
            best = sa_result
            best_count = sa_count
    
    # Final greedy local search pass
    if time.time() < deadline - 2:
        final_result, final_count = local_search(best, deadline - 2, affected)
        if final_count < best_count:
            best = final_result
            best_count = final_count
    
    return best.tolist()
