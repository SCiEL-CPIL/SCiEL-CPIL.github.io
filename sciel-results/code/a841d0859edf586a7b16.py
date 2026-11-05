import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    pos_aps = [[] for _ in range(n)]
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                idx = len(aps)
                aps.append((a - 1, b - 1, c - 1, e - 1))
                for p in (a - 1, b - 1, c - 1, e - 1):
                    pos_aps[p].append(idx)
    
    num_aps = len(aps)
    
    for num_colors in [3, 4, 5, 6]:
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            violations = [0] * num_aps
            total_violations = 0
            for i, ap in enumerate(aps):
                nd = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]})
                if nd < 3:
                    violations[i] = 1
                    total_violations += 1
            
            if total_violations == 0:
                return colors
            
            for iteration in range(80000):
                if time.time() > deadline - 2:
                    break
                
                if total_violations == 0:
                    return colors
                
                ap_idx = random.randrange(num_aps)
                while violations[ap_idx] == 0:
                    ap_idx = random.randrange(num_aps)
                
                ap = aps[ap_idx]
                pos = random.choice(ap)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                
                delta = 0
                for aidx in pos_aps[pos]:
                    a = aps[aidx]
                    c0 = colors[a[0]]
                    c1 = colors[a[1]]
                    c2 = colors[a[2]]
                    c3 = colors[a[3]]
                    old_nd = len({c0, c1, c2, c3})
                    if a[0] == pos:
                        c0 = new_color
                    elif a[1] == pos:
                        c1 = new_color
                    elif a[2] == pos:
                        c2 = new_color
                    elif a[3] == pos:
                        c3 = new_color
                    new_nd = len({c0, c1, c2, c3})
                    delta += (1 if new_nd < 3 else 0) - (1 if old_nd < 3 else 0)
                
                if delta <= 0:
                    colors[pos] = new_color
                    for aidx in pos_aps[pos]:
                        a = aps[aidx]
                        nd = len({colors[a[0]], colors[a[1]], colors[a[2]], colors[a[3]]})
                        violations[aidx] = 1 if nd < 3 else 0
                    total_violations += delta
    
    return [x % 4 for x in range(1, n + 1)]
