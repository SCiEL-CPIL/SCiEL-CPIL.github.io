import time
import itertools

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1
    
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def _build_1_factorization(n):
    """Build a 1-factorization of K_n (n must be even). Returns list of n-1 perfect matchings."""
    matchings = []
    others = list(range(1, n))
    for round_idx in range(n - 1):
        matching = []
        v0 = 0
        v1 = others[round_idx % (n - 1)]
        matching.append((v0, v1))
        remaining = [others[(round_idx + 1 + i) % (n - 1)] for i in range(n - 2)]
        for i in range(len(remaining) // 2):
            matching.append((remaining[i], remaining[len(remaining) - 1 - i]))
        matchings.append(matching)
    return matchings

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    best_colors = None
    best_count = 999
    
    # Build 1-factorization of K16
    matchings = _build_1_factorization(n)
    
    # Assign color = matching index to each edge
    colors_1f = [0] * (n * (n - 1) // 2)
    for m_idx, matching in enumerate(matchings):
        for (u, v) in matching:
            if u > v:
                u, v = v, u
            # Position in lexicographic order
            pos = u * (n - 1) - u * (u - 1) // 2 + (v - 1 - u)
            colors_1f[pos] = m_idx
    
    if _check_k4_fast(colors_1f):
        count = _num_colours(colors_1f)
        if count < best_count:
            best_count = count
            best_colors = colors_1f[:]
    
    # Try to merge colors to reduce count
    if best_colors is not None:
        for _ in range(5):  # Try up to 5 rounds of merging
            if time.time() > deadline - 15:
                break
            
            unique = sorted(set(best_colors))
            if len(unique) <= best_count:
                break
            
            merged = False
            for c1, c2 in itertools.combinations(unique, 2):
                if time.time() > deadline - 10:
                    break
                
                new_colors = [c1 if c == c2 else c for c in best_colors]
                if _check_k4_fast(new_colors):
                    best_colors = new_colors
                    best_count = _num_colours(best_colors)
                    merged = True
                    break
            
            if not merged:
                break
    
    # If no valid coloring found, fallback
    if best_colors is None:
        import random
        while time.time() < deadline - 5:
            best_colors = [random.randint(0, 16) for _ in range(n * (n - 1) // 2)]
            if _check_k4_fast(best_colors):
                break
    
    return best_colors
