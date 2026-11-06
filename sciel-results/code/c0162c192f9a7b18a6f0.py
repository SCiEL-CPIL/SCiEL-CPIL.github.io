def construct():
    import itertools
    
    def check_validity(colors, n=16):
        M = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                M[i][j] = colors[idx]
                M[j][i] = colors[idx]
                idx += 1
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        distinct = {M[a][b], M[a][c], M[a][d], M[b][c], M[b][d], M[c][d]}
                        if len(distinct) < 5:
                            return False
        return True

    def get_colors(values, modulus):
        n = 16
        return [(values[i] * values[j]) % modulus for i in range(n) for j in range(i+1, n)]

    best_colors = None
    best_count = 100
    
    # Try several moduli and value sets
    # We want to minimize the number of distinct colors used.
    
    candidates = []
    
    # Modulus 19
    for start in range(1, 4):
        vals = list(range(start, start+16))
        # Ensure values are non-zero mod 19
        vals = [v % 19 for v in vals]
        if 0 in vals:
            continue
        candidates.append((vals, 19))
        
    # Powers of primitive root in Z_19 (2 is primitive)
    vals_pow = [(2**i) % 19 for i in range(16)]
    candidates.append((vals_pow, 19))
    
    # Modulus 23
    for start in range(1, 4):
        vals = list(range(start, start+16))
        vals = [v % 23 for v in vals]
        if 0 in vals:
            continue
        candidates.append((vals, 23))
        
    # Powers of primitive root in Z_23 (5 is primitive)
    vals_pow23 = [(5**i) % 23 for i in range(16)]
    candidates.append((vals_pow23, 23))

    # Modulus 17 (Multiplicative) - We know this is invalid for full set, but maybe a subset?
    # No, we need 16 distinct values. Z_17* has 16. So it's the full set. Invalid.
    
    for vals, mod in candidates:
        cols = get_colors(vals, mod)
        if check_validity(cols):
            num_colors = len(set(cols))
            if num_colors < best_count:
                best_count = num_colors
                best_colors = cols
                
    if best_colors is not None:
        return best_colors
        
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
