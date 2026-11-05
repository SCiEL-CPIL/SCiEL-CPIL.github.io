import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256
ALL_MASK = (1 << M) - 1

# ---------- Precompute COVER and PAIR_COVERS ----------
def _build_cover_and_pair_covers():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    pair_covers = {}
    
    # Initialize pair_covers with empty masks
    # We can use a dict of ints
    
    for p in range(M):
        d_pa = D2[p, A][:, None]   # M x 1
        d_pb = D2[p, A][None, :]   # 1 x M
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        pairs = list(zip(a[sel].tolist(), b[sel].tolist()))
        cover.append(pairs)
        
        # Update pair_covers
        for a, b in pairs:
            if (a, b) not in pair_covers:
                pair_covers[(a, b)] = 0
            pair_covers[(a, b)] |= (1 << p)
            
    return cover, pair_covers

COVER, PAIR_COVERS = _build_cover_and_pair_covers()

def _is_maximal_fast(S):
    S_list = list(S)
    if not S_list:
        return False
    covered_mask = 0
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            key = (a, b) if a < b else (b, a)
            if key in PAIR_COVERS:
                covered_mask |= PAIR_COVERS[key]
    
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
        
    unchecked = ALL_MASK & ~S_mask
    if not unchecked:
        return True
        
    u = unchecked
    while u:
        p = (u & -u).bit_length() - 1
        u &= u - 1
        if not (covered_mask >> p) & 1:
            return False
    return True

def _shrink_fast(S):
    S = set(S)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            S.discard(x)
            if _is_maximal_fast(S):
                improved = True
            else:
                S.add(x)
    return S

def _greedy_cover_randomized():
    S_mask = 0
    S_list = []
    C_S = 0
    U = ALL_MASK
    
    while U:
        cand_mask = ALL_MASK & ~(S_mask | C_S)
        if not cand_mask:
            return None
        
        max_count = -1
        best_candidates = []
        
        tmp_cand = cand_mask
        while tmp_cand:
            v = (tmp_cand & -tmp_cand).bit_length() - 1
            tmp_cand &= tmp_cand - 1
            
            new_cover = 0
            for a in S_list:
                key = (a, v) if a < v else (v, a)
                if key in PAIR_COVERS:
                    new_cover |= PAIR_COVERS[key]
            
            new_cover &= U
            count = new_cover.bit_count()
            
            if count > max_count:
                max_count = count
                best_candidates = [(v, new_cover)]
            elif count == max_count and count > 0:
                best_candidates.append((v, new_cover))
        
        if not best_candidates or max_count == 0:
            return None
            
        v, best_new_cover_mask = random.choice(best_candidates)
        
        S_mask |= (1 << v)
        S_list.append(v)
        
        full_new_cover = 0
        for a in S_list[:-1]:
            key = (a, v) if a < v else (v, a)
            if key in PAIR_COVERS:
                full_new_cover |= PAIR_COVERS[key]
        
        C_S |= full_new_cover
        U &= ~best_new_cover_mask
        
    return set(S_list)

def construct():
    t0 = time.time()
    deadline = t0 + 285.0
    
    best = None
    best_size = M + 1
    
    for _ in range(100):
        if time.time() > deadline:
            break
        S = _greedy_cover_randomized()
        if S:
            S = _shrink_fast(S)
            if len(S) < best_size:
                best = S
                best_size = len(S)
                
    if best is None:
        # Fallback: return a random valid set? Or empty?
        # The task requires a maximal set.
        # If greedy fails, we can try a simple greedy without coverage optimization, just compatibility.
        # But greedy_cover_randomized should rarely fail if we have enough candidates.
        # Let's just return an empty list if it fails, which might be invalid, but better than timeout.
        # Actually, let's try one more simple greedy.
        S = set()
        inS = [False]*M
        order = list(range(M))
        random.shuffle(order)
        for v in order:
            # Check compatibility
            comp = True
            for a, b in COVER[v]:
                if inS[a] and inS[b]:
                    comp = False
                    break
            if comp:
                S.add(v)
                inS[v] = True
        S = _shrink_fast(S)
        if len(S) < best_size:
            best = S
            best_size = len(S)

    return [POINTS[i] for i in sorted(best)] if best else []
