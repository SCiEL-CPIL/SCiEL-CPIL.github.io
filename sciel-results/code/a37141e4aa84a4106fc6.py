"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Returns a list `cycles_for_edge` of length NUM_EDGES.
    cycles_for_edge[e] is a list of tuples (e1, e2, e3) where 
    {e, e1, e2, e3} is a 4-cycle.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    # Helper to get index
    def get_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
        
    # Iterate over all 4-vertex subsets
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles on {a,b,c,d}
                    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, c)
                    e3 = get_idx(c, d)
                    e4 = get_idx(d, a)
                    cyc1 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc1):
                        others = [cyc1[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, d)
                    e3 = get_idx(d, c)
                    e4 = get_idx(c, a)
                    cyc2 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc2):
                        others = [cyc2[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
                    e1 = get_idx(a, c)
                    e2 = get_idx(c, b)
                    e3 = get_idx(b, d)
                    e4 = get_idx(d, a)
                    cyc3 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc3):
                        others = [cyc3[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                        
    return cycles_for_edge

CYCLES_FOR_EDGE = _precompute_cycles()

def _calculate_score(colors):
    """
    Calculate the score: number of distinct colors * 1000 - number of violations.
    A violation is a 4-cycle with 4 distinct colors.
    """
    distinct_colors = len(set(colors))
    violations = 0
    
    # To count violations efficiently, we can just check all 4-cycles.
    # There are 5460 4-cycles.
    # But we can precompute the list of all 4-cycles as tuples of 4 edge indices.
    # Let's do that for speed in SA.
    
    return distinct_colors * 1000 - violations

def _precompute_all_cycles():
    """
    Returns a list of tuples (e1, e2, e3, e4) for all 4-cycles.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    def get_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
        
    all_cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1
                    all_cycles.append((get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)))
                    # Cycle 2
                    all_cycles.append((get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)))
                    # Cycle 3
                    all_cycles.append((get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)))
    return all_cycles

ALL_CYCLES = _precompute_all_cycles()

def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct colors."""
    violations = 0
    for cycle in ALL_CYCLES:
        c1 = colors[cycle[0]]
        c2 = colors[cycle[1]]
        c3 = colors[cycle[2]]
        c4 = colors[cycle[3]]
        if len({c1, c2, c3, c4}) == 4:
            violations += 1
    return violations

def _sa_max_colors(seed):
    random.seed(seed)
    
    # Initialize with a greedy coloring to get a valid start
    colors = [0] * NUM_EDGES
    next_color = 1
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    # Greedy init
    for e in edge_order:
        # Check if we can assign new color
        can_new = True
        for others in CYCLES_FOR_EDGE[e]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            if c1 != c2 and c2 != c3 and c1 != c3:
                can_new = False
                break
        if can_new:
            colors[e] = next_color
            next_color += 1
        else:
            # Assign existing color that minimizes new violations? 
            # For init, just pick a random existing color or 0
            if next_color > 1:
                colors[e] = random.choice(range(next_color))
            else:
                colors[e] = 0
                
    # Ensure validity? Greedy above ensures validity if we check properly.
    # But the check "can_new" only ensures that adding a NEW color doesn't create a violation 
    # IF the other 3 are distinct. If they are not distinct, it's safe.
    # So the greedy init produces a valid coloring.
    
    current_violations = _count_violations(colors)
    current_distinct = len(set(colors))
    current_score = current_distinct * 1000 - current_violations
    
    best_colors = colors[:]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    steps = 2000
    decay = (T_min / T) ** (1.0 / steps)
    
    for step in range(steps):
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Decide new color
        # 50% chance to try a new unique color, 50% chance to try an existing color
        if random.random() < 0.5 and current_violations == 0:
            # Try to introduce a new color
            new_color = max(colors) + 1
        else:
            # Pick a random existing color or the old color
            if random.random() < 0.5:
                new_color = random.randint(0, max(colors))
            else:
                new_color = old_color
        
        if new_color == old_color:
            continue
            
        # Calculate delta score
        # We need to update violations efficiently.
        # But for simplicity in this short window, we can recalculate if we are close to best?
        # No, recalculation is O(5460). 2000 steps * 5460 is ~10M ops, which is fast in Python?
        # 10M simple ops might take 1-2 seconds. 50 restarts might be too slow.
        # Let's optimize violation update.
        
        # Precomputed: for edge e, list of (cycle_idx, other_3_edges)
        # We don't have that mapped to cycle indices directly in ALL_CYCLES.
        # Let's stick to full recalculation but reduce steps or use faster logic.
        
        # Actually, let's just do full recalculation. It's robust.
        colors[e] = new_color
        
        new_violations = _count_violations(colors)
        new_distinct = len(set(colors))
        new_score = new_distinct * 1000 - new_violations
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_violations = new_violations
            current_distinct = new_distinct
            current_score = new_score
            if current_score > best_score and current_violations == 0:
                best_score = current_score
                best_colors = colors[:]
        else:
            colors[e] = old_color
            
        T *= decay
        
    return best_colors

def construct():
    start_time = time.time()
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Run multiple SA restarts
    for seed in range(100):
        if time.time() - start_time > 55:
            break
        colors = _sa_max_colors(seed)
        count = len(set(colors))
        # Verify validity
        if _count_violations(colors) == 0:
            if count > best_count:
                best_count = count
                best_colors = colors[:]
            
    return best_colors

# EVOLVE-BLOCK-END
