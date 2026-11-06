import time

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    n = 128
    
    # Precompute all nonconstant 4-term APs
    aps = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, which APs involve it?
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    def check_pos_valid(colors, pos):
        for idx in pos_aps[pos]:
            a, b, c, d = aps[idx]
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    # Greedy construction
    colors = [0] * n
    for i in range(n):
        found = False
        for c in range(128):
            colors[i] = c
            if check_pos_valid(colors, i):
                found = True
                break
        if not found:
            colors[i] = 0
    
    # Local search to reduce colors
    used = set(colors)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(used) <= 3:
            break
        used_list = sorted(used)
        merged = False
        for i in range(len(used_list)):
            for j in range(i + 1, len(used_list)):
                ca, cb = used_list[i], used_list[j]
                old = [x if x != ca else cb for x in colors]
                valid = True
                for a, b, c, d in aps:
                    if len({old[a], old[b], old[c], old[d]}) < 3:
                        valid = False
                        break
                if valid:
                    colors = old
                    used.discard(ca)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    return colors
