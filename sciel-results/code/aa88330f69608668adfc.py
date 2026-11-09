import time
import random

def count_mono_4ap(colors):
    n = 256
    c = colors
    count = 0
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i = a-1
            col = c[i]
            if col == c[i+d] == c[i+2*d] == c[i+3*d]:
                count += 1
    return count

def thue_morse(n):
    seq = [0]*n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_affected(n=256):
    """For each position p (0-indexed), store list of (d, a) such that p is in the 4-AP a, a+d, a+2d, a+3d"""
    affected = [[] for _ in range(n)]
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a-1
            for j in range(4):
                p = i0 + j*d
                if p < n:
                    affected[p].append((d, a))
    return affected

def local_search(colors, deadline, affected):
    c = list(colors)
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            old = c[p]
            c[p] = 1 - old
            new = cur
            for (d, a) in affected[p]:
                i0 = a - 1
                idxs = [i0, i0+d, i0+2*d, i0+3*d]
                was_mono = all(c[idx] == old for idx in idxs)
                is_mono = all(c[idx] == (1-old) for idx in idxs)
                if was_mono and not is_mono:
                    new -= 1
                elif not was_mono and is_mono:
                    new += 1
            if new < cur:
                cur = new
                improved = True
            else:
                c[p] = old
    return c, cur

def simulated_annealing(colors, deadline, affected):
    c = list(colors)
    cur = count_mono_4ap(c)
    best_c = list(c)
    best_val = cur
    
    t_start = time.time()
    total_time = deadline - t_start
    if total_time <= 0:
        return best_c, best_val
    
    # Temperature schedule
    T0 = 5.0
    T_end = 0.01
    # We'll do as many iterations as we can
    # Estimate: each iteration ~ 50-100us with incremental counting
    # Let's just run until deadline
    
    import math
    # Number of iterations we estimate we can do
    # Be conservative: assume 200us per iteration
    max_iters = int(total_time * 5000)  # ~5000 iters/sec
    if max_iters < 100:
        max_iters = 100
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        # Cool temperature
        frac = it / max_iters
        T = T0 * (T_end / T0) ** frac
        
        p = random.randint(0, 255)
        old = c[p]
        c[p] = 1 - old
        
        new = cur
        for (d, a) in affected[p]:
            i0 = a - 1
            idxs = [i0, i0+d, i0+2*d, i0+3*d]
            was_mono = all(c[idx] == old for idx in idxs)
            is_mono = all(c[idx] == (1-old) for idx in idxs)
            if was_mono and not is_mono:
                new -= 1
            elif not was_mono and is_mono:
                new += 1
        
        delta = new - cur
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur = new
            if cur < best_val:
                best_val = cur
                best_c = list(c)
        else:
            c[p] = old
    
    return best_c, best_val

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Precompute affected structures
    affected = build_affected()
    
    candidates = []
    candidates.append(thue_morse(256))
    candidates.append([(i//2)%2 for i in range(256)])
    candidates.append([(i//3)%2 for i in range(256)])
    candidates.append([(i//4)%2 for i in range(256)])
    tm_full = thue_morse(257)
    candidates.append(tm_full[1:257])
    def v2_color(n):
        out = []
        for i in range(1, n+1):
            x = i
            p = 0
            while x % 2 == 0:
                x //= 2
                p += 1
            out.append(p % 2)
        return out
    candidates.append(v2_color(256))
    candidates.append([(i % 2) ^ ((i//2) % 2) for i in range(256)])
    
    best = None
    best_count = float('inf')
    for cand in candidates:
        if time.time() > deadline:
            break
        cnt = count_mono_4ap(cand)
        if cnt < best_count:
            best_count = cnt
            best = list(cand)
    
    if best is None:
        best = list(candidates[0])
        best_count = count_mono_4ap(best)
    
    # Phase 1: Greedy local search
    phase1_deadline = start + 40
    best, best_count = local_search(best, phase1_deadline, affected)
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 5:
        sa_deadline = deadline - 2
        sa_result, sa_count = simulated_annealing(best, sa_deadline, affected)
        if sa_count < best_count:
            best = sa_result
            best_count = sa_count
    
    # Phase 3: Final greedy polish
    if time.time() < deadline - 3:
        best, best_count = local_search(best, deadline - 1, affected)
    
    return best
