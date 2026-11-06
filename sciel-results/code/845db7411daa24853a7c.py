"""Rewritten program: local search maximizing distinct edge colors in K16 without rainbow C4."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
                    e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
                    e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_TO_CYCLES = [[] for _ in range(M)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_TO_CYCLES[e].append(ci)

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_rainbow(cyc, colors):
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return len({c0, c1, c2, c3}) == 4

def make_structured(mode, rng):
    colors = [0] * M
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if mode == 0:
                colors[idx] = (i + j) % 120
            elif mode == 1:
                colors[idx] = (j - i) % 120
            elif mode == 2:
                colors[idx] = (i * j) % 120
            elif mode == 3:
                colors[idx] = rng.randint(0, 119)
            elif mode == 4:
                colors[idx] = ((i + j) % 120 + rng.randint(-10, 10)) % 120
            else:
                colors[idx] = ((i * 7 + j * 13) % 120 + rng.randint(-5, 5)) % 120
            idx += 1
    return colors

def construct():
    deadline = time.time() + 85
    rng = random.Random(777)
    
    best_colors = None
    best_score = 0
    
    # Try structured initializations
    for mode in range(6):
        if time.time() > deadline - 10:
            break
        colors = make_structured(mode, rng)
        if is_valid(colors):
            score = count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    if best_colors is None:
        return [0] * M
    
    colors = best_colors[:]
    rainbow = [check_rainbow(c, colors) for c in CYCLES]
    num_rainbow = sum(rainbow)
    
    # Local search: try random color changes that don't create rainbows
    for it in range(10000000):
        if time.time() > deadline - 3:
            break
        
        edge = rng.randrange(M)
        old_color = colors[edge]
        new_color = rng.randint(0, 119)
        if new_color == old_color:
            continue
        
        affected = EDGE_TO_CYCLES[edge]
        
        ok = True
        for aci in affected:
            ac = CYCLES[aci]
            c0 = colors[ac[0]] if ac[0] != edge else new_color
            c1 = colors[ac[1]] if ac[1] != edge else new_color
            c2 = colors[ac[2]] if ac[2] != edge else new_color
            c3 = colors[ac[3]] if ac[3] != edge else new_color
            if len({c0, c1, c2, c3}) == 4:
                if not rainbow[aci]:
                    ok = False
                    break
        
        if not ok:
            continue
        
        colors[edge] = new_color
        delta = 0
        for aci in affected:
            ac = CYCLES[aci]
            was = rainbow[aci]
            is_r = check_rainbow(ac, colors)
            if was and not is_r:
                delta -= 1
            elif not was and is_r:
                delta += 1
            rainbow[aci] = is_r
        num_rainbow += delta
        
        if num_rainbow > 0:
            colors[edge] = old_color
            for aci in affected:
                ac = CYCLES[aci]
                is_r = check_rainbow(ac, colors)
                if rainbow[aci] != is_r:
                    if is_r:
                        num_rainbow += 1
                    else:
                        num_rainbow -= 1
                    rainbow[aci] = is_r
            continue
        
        new_score = count_colors(colors)
        if new_score > best_score:
            best_score = new_score
            best_colors = colors[:]
    
    return best_colors
