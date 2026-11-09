def construct():
    n = 128
    
    def try_k_colors(k, look_depth=2):
        colors = [-1] * n
        
        def check_valid(pos, color):
            x = pos + 1
            for d in range(1, (x - 1) // 3 + 1):
                i0 = x - 1 - 3 * d
                i1 = i0 + d
                i2 = i0 + 2 * d
                if i0 < 0:
                    continue
                if colors[i0] == -1 or colors[i1] == -1 or colors[i2] == -1:
                    continue
                if len({colors[i0], colors[i1], colors[i2], color}) < 3:
                    return False
            return True
        
        def lookahead(pos, depth):
            if depth >= look_depth:
                return True
            np_ = pos + 1
            if np_ >= n:
                return True
            for c in range(k):
                if not check_valid(np_, c):
                    continue
                colors[np_] = c
                if lookahead(np_, depth + 1):
                    colors[np_] = -1
                    return True
                colors[np_] = -1
            return False
        
        for pos in range(n):
            placed = False
            for c in range(k):
                if not check_valid(pos, c):
                    continue
                colors[pos] = c
                if lookahead(pos, 0):
                    placed = True
                    break
                colors[pos] = -1
            if not placed:
                for c in range(k):
                    if check_valid(pos, c):
                        colors[pos] = c
                        break
            if colors[pos] == -1:
                return None
        
        # Final verification
        for a in range(1, n + 1):
            for d in range(1, (n - a + 1) // 3):
                if a + 3 * d > n:
                    break
                vals = [colors[a - 1], colors[a + d - 1], colors[a + 2 * d - 1], colors[a + 3 * d - 1]]
                if len(set(vals)) < 3:
                    return None
        return colors
    
    for k in range(3, 129):
        res = try_k_colors(k, look_depth=2)
        if res is not None:
            return res
    
    # Fallback greedy
    colors = []
    for x in range(1, n + 1):
        for c in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                i0 = x - 1 - 3 * d
                if i0 < 0:
                    continue
                prev = [colors[i0], colors[i0 + d], colors[i0 + 2 * d]]
                if len(set(prev + [c])) < 3:
                    ok = False
                    break
            if ok:
                colors.append(c)
                break
    return colors
