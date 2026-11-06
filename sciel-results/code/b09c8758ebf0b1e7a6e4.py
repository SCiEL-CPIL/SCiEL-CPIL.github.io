import time
import random

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    def is_valid(coloring):
        """Check if coloring is valid for all 4-APs."""
        n = len(coloring)
        # Precompute all 4-APs? No, too many.
        # Check on the fly for a specific change?
        # For initial check, we can iterate.
        # Number of 4-APs in 1..128:
        # d=1: 125, d=2: 122, ... d=42: 1.
        # Total ~ sum(128 - 3d) for d=1..42.
        # Approx 42 * 100 = 4200 APs.
        # Checking 4200 APs is fast.
        
        for d in range(1, 43):
            for a in range(1, 129 - 3*d):
                c0 = coloring[a-1]
                c1 = coloring[a+d-1]
                c2 = coloring[a+2*d-1]
                c3 = coloring[a+3*d-1]
                if len({c0, c1, c2, c3}) < 3:
                    return False
        return True

    def count_colors(coloring):
        return len(set(coloring))

    # 1. Greedy Construction (Seed Logic)
    def greedy_construct():
        colors = []
        for x in range(1, 129):
            for color in range(128):
                valid = True
                # Check all 4-APs ending at x
                max_d = (x - 1) // 3
                for d in range(1, max_d + 1):
                    # Indices: x-1, x-1-d, x-1-2d, x-1-3d
                    # Colors: color, colors[x-2-d], colors[x-1-2d], colors[x-2-3d]
                    # Wait, x is 1-based.
                    # pos 0: x-1
                    # pos 1: x-1-d
                    # pos 2: x-1-2d
                    # pos 3: x-1-3d
                    c0 = color
                    c1 = colors[x-2-d] if x-2-d >= 0 else None
                    c2 = colors[x-1-2*d] if x-1-2*d >= 0 else None
                    c3 = colors[x-2-3*d] if x-2-3*d >= 0 else None
                    
                    # Only check if all previous indices are valid (which they are for d <= (x-1)//3)
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
        return colors

    # 2. Modular Construction
    def modular_construct(m):
        return [(i) % m for i in range(128)]

    best_colors = None
    best_count = 128

    # Try Greedy
    if time.time() - start_time < deadline:
        g_colors = greedy_construct()
        g_count = count_colors(g_colors)
        if g_count < best_count:
            best_count = g_count
            best_colors = g_colors

    # Try Modular m=43
    if time.time() - start_time < deadline:
        m_colors = modular_construct(43)
        m_count = count_colors(m_colors)
        if m_count < best_count:
            best_count = m_count
            best_colors = m_colors

    # Try Modular m=42? No, fails.
    # Try Modular m=44, 45? No, 43 is smaller.

    # Local Search / Compression
    if best_colors is not None and time.time() - start_time < deadline:
        current = best_colors[:]
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            # Try to reduce color at each position
            for i in range(128):
                if time.time() - start_time > deadline - 5:
                    break
                orig_color = current[i]
                # Try colors smaller than orig_color
                for c in range(orig_color):
                    current[i] = c
                    # Check validity of APs involving i
                    # i is 0-indexed, so pos = i+1
                    pos = i + 1
                    valid = True
                    # i can be part of AP as a, a+d, a+2d, a+3d
                    # 1. i is a: a=i, a+d, a+2d, a+3d. Need pos+3d <= 128.
                    for d in range(1, (128 - pos) // 3 + 1):
                        if len({current[pos-1], current[pos+d-1], current[pos+2*d-1], current[pos+3*d-1]}) < 3:
                            valid = False
                            break
                    if valid:
                        # 2. i is a+d: a=i-d, a+d=i, a+2d=i+d, a+3d=i+2d.
                        # Need a>=1 => i-d >= 1 => d <= i-1.
                        # And i+2d <= 128 => 2d <= 128-i => d <= (128-i)//2.
                        max_d = min(i-1, (128-i)//2)
                        for d in range(1, max_d + 1):
                            if len({current[pos-d-1], current[pos-1], current[pos+d-1], current[pos+2*d-1]}) < 3:
                                valid = False
                                break
                    if valid:
                        # 3. i is a+2d: a=i-2d, a+d=i-d, a+2d=i, a+3d=i+d.
                        # Need a>=1 => 2d <= i-1 => d <= (i-1)//2.
                        # And i+d <= 128 => d <= 128-i.
                        max_d = min((i-1)//2, 128-i)
                        for d in range(1, max_d + 1):
                            if len({current[pos-2*d-1], current[pos-d-1], current[pos-1], current[pos+d-1]}) < 3:
                                valid = False
                                break
                    if valid:
                        # 4. i is a+3d: a=i-3d, a+d=i-2d, a+2d=i-d, a+3d=i.
                        # Need a>=1 => 3d <= i-1 => d <= (i-1)//3.
                        max_d = (i-1)//3
                        for d in range(1, max_d + 1):
                            if len({current[pos-3*d-1], current[pos-2*d-1], current[pos-d-1], current[pos-1]}) < 3:
                                valid = False
                                break
                    
                    if valid:
                        improved = True
                        break
                    else:
                        current[i] = orig_color
            
            # Compress colors
            new_colors = sorted(set(current))
            mapping = {old: new for new, old in enumerate(new_colors)}
            if len(new_colors) < best_count:
                best_count = len(new_colors)
                current = [mapping[c] for c in current]
                best_colors = current[:]

    return best_colors

