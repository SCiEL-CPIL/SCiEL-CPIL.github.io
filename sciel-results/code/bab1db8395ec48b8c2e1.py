import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    all_4cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        all_4cycles.append((a, b, c, d))
        all_4cycles.append((a, b, d, c))
        all_4cycles.append((a, c, b, d))
    
    # Map edge to list of cycle indices
    edge_to_cycles = {}
    for idx, (v1, v2, v3, v4) in enumerate(all_4cycles):
        for e in [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(idx)
    
    def run_greedy(seed):
        random.seed(seed)
        color = {}
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            key = (u, v)
            for c in range(num_colors):
                valid = True
                for ci in edge_to_cycles.get(key, []):
                    v1, v2, v3, v4 = all_4cycles[ci]
                    ce = [(v1,v2), (v2,v3), (v3,v4), (v4,v1)]
                    ok = True
                    ocols = []
                    for e in ce:
                        ek = (min(e[0],e[1]), max(e[0],e[1]))
                        if ek == key:
                            continue
                        if ek not in color:
                            ok = False
                            break
                        ocols.append(color[ek])
                    if ok:
                        if c in ocols:
                            valid = False
                            break
                if valid:
                    color[key] = c
                    break
        return color
    
    best_color = None
    best_count = 0
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        color = run_greedy(seed)
        if len(color) > best_count:
            best_count = len(color)
            best_color = color
    
    if best_color is None:
        return [[0, v, 0] for v in range(1, n)]
    
    return [[u, v, c] for (u, v), c in best_color.items()]
