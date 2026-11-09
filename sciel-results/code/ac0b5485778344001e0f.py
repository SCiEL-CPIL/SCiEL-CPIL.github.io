import time

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

def local_search(colors, deadline):
    c = list(colors)
    cur = count_mono_4ap(c)
    affected = build_affected()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            old = c[p]
            new_val = 1 - old
            c[p] = new_val
            # Compute change in mono count by checking only affected 4-APs
            delta = 0
            for (d, a) in affected[p]:
                i0 = a - 1
                # Current state has p flipped
                is_mono_now = (c[i0] == c[i0+d] == c[i0+2*d] == c[i0+3*d])
                # Revert to check old state
                c[p] = old
                was_mono = (c[i0] == c[i0+d] == c[i0+2*d] == c[i0+3*d])
                c[p] = new_val
                if is_mono_now and not was_mono:
                    delta += 1
                elif was_mono and not is_mono_now:
                    delta -= 1
            new_cur = cur + delta
            if new_cur < cur:
                cur = new_cur
                improved = True
            else:
                c[p] = old
    return c, cur

def construct():
    start = time.time()
    deadline = start + 140
    
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
    # Add shifted Thue-Morse variants
    tm2 = thue_morse(260)
    candidates.append(tm2[2:258])
    candidates.append(tm2[4:260])
    # Add complement of Thue-Morse
    candidates.append([1-x for x in thue_morse(256)])
    # Add XOR with higher bit patterns
    candidates.append([(thue_morse(256)[i]) ^ ((i>>2)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>3)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>4)&1) for i in range(256)])
    
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
    
    # Local search from best candidate
    best, best_count = local_search(best, deadline)
    
    return best
