import time
import random
from itertools import combinations, permutations

def get_edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return sum(n - 1 - k for k in range(i)) + (j - i - 1)

def precompute_cycles(n):
    """Precompute all 6-cycles and build edge-to-cycle index map."""
    base_cycles = set()
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
    
    cycles = []
    # edge_to_cycles[e] = list of cycle indices that use edge e
    edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
    
    for comb in combinations(range(n), 6):
        mapping = {k: v for k, v in enumerate(comb)}
        for b_cycle in base_cycles:
            mapped_cycle = tuple(mapping[v] for v in b_cycle)
            edges = []
            for k in range(6):
                u = mapped_cycle[k]
                v = mapped_cycle[(k + 1) % 6]
                idx = get_edge_index(u, v, n)
                edges.append(idx)
            cycle_idx = len(cycles)
            cycles.append(tuple(edges))
            for e in edges:
                edge_to_cycles[e].append(cycle_idx)
    
    return cycles, edge_to_cycles

def construct():
    start_time = time.time()
    deadline = 140.0  # Keep safety margin before 150s kill
    
    n = 17
    num_edges = n * (n - 1) // 2
    cycles, edge_to_cycles = precompute_cycles(n)
    num_cycles = len(cycles)
    
    # For each cycle, store edge indices as a list for fast access
    cycle_edges = [list(c) for c in cycles]
    
    best_penalty = float('inf')
    best_colors = None
    
    def compute_penalty(colors):
        """Count monochromatic 6-cycles."""
        p = 0
        for ce in cycle_edges:
            c0 = colors[ce[0]]
            if (colors[ce[1]] == c0 and colors[ce[2]] == c0 and 
                colors[ce[3]] == c0 and colors[ce[4]] == c0 and colors[ce[5]] == c0):
                p += 1
        return p
    
    def try_improve(colors, cur_penalty):
        """Try to improve by changing one edge. Returns (new_penalty, changed_edge, new_color) or None."""
        # Try each edge, pick the one with best improvement
        best_imp = None
        best_new_p = cur_penalty
        
        for idx in range(num_edges):
            old_color = colors[idx]
            affected = edge_to_cycles[idx]
            if not affected:
                continue
            
            for c in range(4):
                if c == old_color:
                    continue
                # Compute delta: how many cycles become/stop being monochromatic
                delta = 0
                for ci in affected:
                    ce = cycle_edges[ci]
                    # Check if this cycle is monochromatic with old_color
                    is_mono_old = True
                    for e in ce:
                        if e == idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    # Check if this cycle would be monochromatic with new color c
                    is_mono_new = True
                    for e in ce:
                        if e == idx:
                            continue
                        if colors[e] != c:
                            is_mono_new = False
                            break
                    if is_mono_old and not is_mono_new:
                        delta -= 1
                    elif not is_mono_old and is_mono_new:
                        delta += 1
                
                new_p = cur_penalty + delta
                if new_p < best_new_p:
                    best_new_p = new_p
                    best_imp = (idx, c, delta)
        
        return best_imp, best_new_p
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        cur_penalty = compute_penalty(colors)
        
        if cur_penalty < best_penalty:
            best_penalty = cur_penalty
            best_colors = colors[:]
        
        if cur_penalty == 0:
            break
        
        # Stochastic local search with occasional random moves
        max_steps = 500
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            affected = edge_to_cycles[idx]
            
            if not affected:
                continue
            
            # Find best color for this edge
            best_c = old_color
            best_delta = 0
            
            for c in range(4):
                if c == old_color:
                    continue
                delta = 0
                for ci in affected:
                    ce = cycle_edges[ci]
                    is_mono_old = True
                    for e in ce:
                        if e == idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    is_mono_new = True
                    for e in ce:
                        if e == idx:
                            continue
                        if colors[e] != c:
                            is_mono_new = False
                            break
                    if is_mono_old and not is_mono_new:
                        delta -= 1
                    elif not is_mono_old and is_mono_new:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_c = c
            
            if best_delta < 0:
                # Accept improving move
                colors[idx] = best_c
                cur_penalty += best_delta
                if cur_penalty < best_penalty:
                    best_penalty = cur_penalty
                    best_colors = colors[:]
                if cur_penalty == 0:
                    break
            elif best_delta == 0:
                # Accept neutral move with some probability (simulated annealing lite)
                if random.random() < 0.3:
                    colors[idx] = best_c
            # else: reject worsening moves (mostly)
            # Occasionally accept worsening to escape local minima
            elif random.random() < 0.05:
                colors[idx] = best_c
                cur_penalty += best_delta
        
        if cur_penalty == 0:
            break
    
    if best_penalty == 0:
        return {'n': n, 'colors': best_colors}
    
    # Fallback: try smaller n if 17 failed
    for try_n in [16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
        tn = try_n
        tn_edges = tn * (tn - 1) // 2
        # Simple random search for smaller n
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            tc = [random.randint(0, 3) for _ in range(tn_edges)]
            # Quick check
            ok = True
            # For smaller n, just do a basic check
            from itertools import combinations as comb2
            for six in comb2(range(tn), 6):
                # Check all 60 cycle orderings
                for perm in permutations(six[1:]):
                    cyc = (six[0],) + perm
                    # Only check canonical to avoid duplicates
                    rev = (six[0],) + perm[::-1]
                    if min(cyc, rev) != cyc:
                        continue
                    c0 = tc[get_edge_index(cyc[0], cyc[1], tn)]
                    mono = True
                    for k in range(6):
                        u, v = cyc[k], cyc[(k+1)%6]
                        if tc[get_edge_index(u, v, tn)] != c0:
                            mono = False
                            break
                    if mono:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return {'n': tn, 'colors': tc}
    
    # Final fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
