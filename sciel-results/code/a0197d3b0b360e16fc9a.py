"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Map (i,j) with i<j to index 0..119"""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def get_cycles():
    """Return list of 4-cycles, each as a list of 4 edge indices."""
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        cycles.append(e1)
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        cycles.append(e2)
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]
        cycles.append(e3)
    return cycles

ALL_CYCLES = get_cycles()

def count_colors(colors):
    return len(set(colors))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(ALL_CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    
    rng = random.Random(42)
    
    def greedy_randomized(seed_val):
        local_rng = random.Random(seed_val)
        colors = [0] * NUM_EDGES
        order = list(range(NUM_EDGES))
        local_rng.shuffle(order)
        
        for e in order:
            can_assign = True
            for ci in edge_to_cycles[e]:
                cyc = ALL_CYCLES[ci]
                other_edges = [x for x in cyc if x != e]
                other_colors = [colors[x] for x in other_edges]
                if len(set(other_colors)) == 3:
                    can_assign = False
                    break
            
            if can_assign:
                colors[e] = count_colors(colors)
        
        return colors
    
    # Run many randomized greedy attempts
    for attempt in range(500):
        if time.time() > deadline:
            break
        c = greedy_randomized(attempt)
        cnt = count_colors(c)
        if cnt > best_count:
            best_count = cnt
            best_colors = c[:]
    
    # Backtracking with MRV and new-color-first
    def backtrack_search():
        nonlocal best_colors, best_count
        
        colors = [0] * NUM_EDGES
        used_colors = set([0])
        
        def rec(depth, max_depth):
            nonlocal best_colors, best_count
            
            if time.time() > deadline:
                return
            
            current_count = len(used_colors)
            if current_count > best_count:
                best_count = current_count
                best_colors = colors[:]
            
            if depth >= max_depth:
                return
            
            candidate_edges = []
            for e in range(NUM_EDGES):
                can_new = True
                for ci in edge_to_cycles[e]:
                    cyc = ALL_CYCLES[ci]
                    other_edges = [x for x in cyc if x != e]
                    other_colors = [colors[x] for x in other_edges]
                    if len(set(other_colors)) == 3:
                        can_new = False
                        break
                if can_new:
                    candidate_edges.append(e)
            
            if not candidate_edges:
                return
            
            candidate_edges.sort(key=lambda e: len(edge_to_cycles[e]))
            
            for e in candidate_edges[:8]:
                if time.time() > deadline:
                    return
                
                old_color = colors[e]
                new_color = len(used_colors)
                colors[e] = new_color
                used_colors.add(new_color)
                
                rec(depth + 1, max_depth)
                
                colors[e] = old_color
                used_colors.discard(new_color)
        
        rec(0, 40)
    
    if time.time() < deadline - 5:
        backtrack_search()
    
    # Final local improvement pass
    if time.time() < deadline - 1:
        for iteration in range(100):
            if time.time() > deadline:
                break
            improved = False
            for e in range(NUM_EDGES):
                if time.time() > deadline:
                    break
                can_new = True
                for ci in edge_to_cycles[e]:
                    cyc = ALL_CYCLES[ci]
                    other_edges = [x for x in cyc if x != e]
                    other_colors = [best_colors[x] for x in other_edges]
                    if len(set(other_colors)) == 3:
                        can_new = False
                        break
                if can_new:
                    best_colors[e] = count_colors(best_colors)
                    improved = True
            if not improved:
                break
    
    return best_colors
# EVOLVE-BLOCK-END
