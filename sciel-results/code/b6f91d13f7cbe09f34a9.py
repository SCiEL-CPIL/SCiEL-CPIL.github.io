import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
            
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = edge_idx[(a, b)]
                    e2 = edge_idx[(b, c)]
                    e3 = edge_idx[(c, d)]
                    e4 = edge_idx[(d, a)]
                    cycles.append((e1, e2, e3, e4))
                    
                    e1 = edge_idx[(a, b)]
                    e2 = edge_idx[(b, d)]
                    e3 = edge_idx[(c, d)]
                    e4 = edge_idx[(a, c)]
                    cycles.append((e1, e2, e3, e4))
                    
                    e1 = edge_idx[(a, c)]
                    e2 = edge_idx[(b, c)]
                    e3 = edge_idx[(b, d)]
                    e4 = edge_idx[(d, a)]
                    cycles.append((e1, e2, e3, e4))

    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    colors = [0] * NUM_EDGES
    
    def is_rainbow(cyc):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        return len({c0, c1, c2, c3}) == 4

    bad_cycles = set()
    for i, cyc in enumerate(cycles):
        if is_rainbow(cyc):
            bad_cycles.add(i)
            
    def try_change(e, new_c):
        old_c = colors[e]
        if old_c == new_c:
            return True
        
        affected = edge_to_cycles[e]
        colors[e] = new_c
        
        newly_bad = []
        newly_good = []
        
        for ci in affected:
            cyc = cycles[ci]
            was_bad = ci in bad_cycles
            is_bad_now = is_rainbow(cyc)
            
            if was_bad and not is_bad_now:
                newly_good.append(ci)
            elif not was_bad and is_bad_now:
                newly_bad.append(ci)
        
        if not newly_bad:
            bad_cycles.difference_update(newly_good)
            return True
        else:
            colors[e] = old_c
            return False

    # Initialization
    for _ in range(2000):
        e = random.randint(0, NUM_EDGES - 1)
        c = random.randint(0, 4)
        try_change(e, c)
        
    # Expansion
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        # Try to use a new color
        if random.random() < 0.7:
            c = random.randint(5, 119)
        else:
            c = random.randint(0, 119)
        try_change(e, c)

    return colors
