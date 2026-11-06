import time
import itertools
import random

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
                    # The 6 edges of K4(a,b,c,d) are:
                    # ab, ac, ad, bc, bd, cd
                    # Their colours are:
                    # ab, ac, ea[d], bc, eb[d], ec[d]
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    best_colors = None
    best_count = 999
    
    # Approach 1: Szekeres-Wilson 1-factorization
    # Vertices: 0..14, 15 (representing infinity)
    # Colours: 0..14
    # Factor k (0..14):
    #   Edge (15, k) has colour k.
    #   For i in 0..14, i != k:
    #       j = (2*k - i) % 15
    #       if i < j:
    #           Edge (i, j) has colour k.
    
    colors_sw = [0] * (n * (n - 1) // 2)
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i == 15 or j == 15:
                # One is infinity, one is k
                k = j if i == 15 else i
                colors_sw[edge_idx] = k
            else:
                # Both in 0..14
                # Find k such that i + j == 2k mod 15
                # 2k = (i + j) % 15
                # k = (i + j) * inv(2) mod 15
                # inv(2) mod 15 is 8 because 2*8=16=1 mod 15
                s = (i + j) % 15
                k = (s * 8) % 15
                colors_sw[edge_idx] = k
            edge_idx += 1
            
    if _check_k4_fast(colors_sw):
        count = _num_colours(colors_sw)
        if count < best_count:
            best_count = count
            best_colors = colors_sw[:]
    else:
        # Fallback if Szekeres-Wilson is invalid (should not be)
        # Use additive mod 17
        colors_add = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if _check_k4_fast(colors_add):
            count = _num_colours(colors_add)
            if count < best_count:
                best_count = count
                best_colors = colors_add[:]

    # Approach 2: Greedy Merging
    if best_colors is not None:
        # Try to merge colours until no improvement
        # We can try multiple random orders of merging to escape local minima
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Reset to best found so far
            current_colors = best_colors[:]
            current_count = best_count
            
            # Get unique colours
            unique = list(set(current_colors))
            
            # Create a random permutation of pairs to try
            pairs = list(itertools.combinations(unique, 2))
            random.seed(attempt) # Deterministic for reproducibility in tests, but different per attempt
            random.shuffle(pairs)
            
            improved = True
            while improved:
                if time.time() > deadline - 10:
                    break
                improved = False
                for c1, c2 in pairs:
                    if c1 not in set(current_colors) or c2 not in set(current_colors):
                        continue
                    
                    # Try merging c2 into c1
                    new_colors = [c1 if c == c2 else c for c in current_colors]
                    if _check_k4_fast(new_colors):
                        current_colors = new_colors
                        current_count = _num_colours(current_colors)
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = current_colors[:]
                        improved = True
                        # Update pairs list? Or just break and restart inner loop?
                        # Restarting inner loop is safer
                        break

    # If no valid coloring found, fallback to random
    if best_colors is None:
        while time.time() < deadline - 5:
            best_colors = [random.randint(0, 16) for _ in range(n * (n - 1) // 2)]
            if _check_k4_fast(best_colors):
                break

    return best_colors
