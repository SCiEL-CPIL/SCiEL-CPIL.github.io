import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    colors = [-1] * n
    
    def is_valid(i, c):
        """Check if assigning color c to position i is valid given current colors."""
        for d in range(1, (i // 3) + 1):
            a = i - 3*d
            if a < 0:
                break
            b = i - 2*d
            c2 = i - d
            if len({colors[a], colors[b], colors[c2], c}) < 3:
                return False
        return True
    
    # Greedy forward pass
    for i in range(n):
        if time.time() > deadline:
            break
        found = False
        # Try colors in order (0, 1, 2, ...) to minimize distinct colors
        for c in range(128):
            if is_valid(i, c):
                colors[i] = c
                found = True
                break
        if not found:
            colors[i] = 0
    
    # Local search to reduce colors
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            distinct = sorted(set(colors))
            for c_remove in distinct:
                if time.time() > deadline - 10:
                    break
                if len(distinct) <= 3:
                    break
                for c_target in distinct:
                    if c_remove == c_target:
                        continue
                    # Try remapping c_remove -> c_target
                    # Check all 4-term APs
                    valid = True
                    for i in range(n):
                        if colors[i] != c_remove:
                            continue
                        # Check APs ending at i
                        for d in range(1, (i // 3) + 1):
                            a = i - 3*d
                            if a < 0:
                                break
                            b = i - 2*d
                            c2 = i - d
                            ca = c_target if colors[a] == c_remove else colors[a]
                            cb = c_target if colors[b] == c_remove else colors[b]
                            cc = c_target if colors[c2] == c_remove else colors[c2]
                            if len({ca, cb, cc, c_target}) < 3:
                                valid = False
                                break
                        if not valid:
                            break
                        # Check APs where i is not the last term
                        for d in range(1, n):
                            for offset in range(4):
                                start = i - offset * d
                                if start < 0 or start + 3*d >= n:
                                    continue
                                cols = []
                                for j in range(4):
                                    pos = start + j*d
                                    if pos == i:
                                        cols.append(c_target)
                                    else:
                                        cols.append(colors[pos])
                                if len(set(cols)) < 3:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        for j in range(n):
                            if colors[j] == c_remove:
                                colors[j] = c_target
                        improved = True
                        break
                if improved:
                    break
    
    return colors
