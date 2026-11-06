import time

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Try modular constructions: color[i] = i % m for m from 3 upward
    # A 4-term AP a,a+d,a+2d,a+3d in [1,128] has d in [1, 42]
    # It fails if m|d (1 color) or m|2d (2 colors: ABAB pattern)
    # For m|d: need m > 42
    # For m|2d: need m > 84 (since 2*42=84)
    # So smallest m is 85
    
    # But let's verify by checking all APs
    def check_coloring(colors):
        """colors is list of length 128, colors[i] is color of position i+1"""
        for a in range(1, 129):
            for d in range(1, (128 - a) // 3 + 1):
                if a + 3*d > 128:
                    break
                c0 = colors[a-1]
                c1 = colors[a+d-1]
                c2 = colors[a+2*d-1]
                c3 = colors[a+3*d-1]
                distinct = len({c0, c1, c2, c3})
                if distinct < 3:
                    return False
        return True
    
    # Try modular constructions
    for m in range(3, 129):
        if time.time() > deadline - 10:
            break
        colors = [(i) % m for i in range(1, 129)]
        if check_coloring(colors):
            return colors
    
    # Fallback: greedy backtracking (original approach)
    colors = []
    for x in range(1, 129):
        if time.time() > deadline - 5:
            # Emergency: assign a safe color
            colors.append(0)
            continue
        for color in range(128):
            valid = True
            # Check all 4-term APs ending at x
            for d in range(1, (x-1)//3 + 1):
                if x - 3*d < 1:
                    break
                # AP: x-3d, x-2d, x-d, x
                c0 = colors[x-1-3*d]
                c1 = colors[x-1-2*d]
                c2 = colors[x-1-d]
                distinct_with_new = len({c0, c1, c2, color})
                if distinct_with_new < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen with enough colors
            colors.append(0)
    
    return colors
