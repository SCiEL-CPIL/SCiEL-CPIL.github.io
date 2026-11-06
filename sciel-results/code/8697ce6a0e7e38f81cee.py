def construct():
    n = 128
    colors = [-1] * n
    
    for p in range(n):
        assigned = False
        for c in range(128):
            valid = True
            d = 1
            while p - 3 * d >= 0:
                a = p - 3 * d
                b = p - 2 * d
                c2 = p - d
                col_a = colors[a]
                col_b = colors[b]
                col_c = colors[c2]
                # Check if these 4 colors use at least 3 distinct values
                if len({col_a, col_b, col_c, c}) < 3:
                    valid = False
                    break
                d += 1
            if valid:
                colors[p] = c
                assigned = True
                break
        if not assigned:
            # Fallback (should not happen with 128 available colors)
            colors[p] = 0
    
    return colors
