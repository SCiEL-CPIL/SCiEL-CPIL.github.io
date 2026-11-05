def construct():
    n = 128
    # Try with 3 colors first
    for num_colors in [3, 4, 5]:
        colors = [-1] * n
        def backtrack(pos):
            if pos == n:
                return True
            # Try each color
            for c in range(num_colors):
                colors[pos] = c
                # Check all 4-term APs ending at pos
                valid = True
                for d in range(1, (pos) // 3 + 1):
                    a = pos - 3*d
                    if a < 0:
                        break
                    # AP: a, a+d, a+2d, a+3d = pos
                    c1 = colors[a]
                    c2 = colors[a+d]
                    c3 = colors[a+2*d]
                    c4 = c
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
                if valid:
                    if backtrack(pos + 1):
                        return True
                colors[pos] = -1
            return False
        if backtrack(0):
            return colors
    return None
