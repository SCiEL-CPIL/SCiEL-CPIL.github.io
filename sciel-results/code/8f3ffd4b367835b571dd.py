import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    def is_valid_assignment(colors, edge, color):
        """Check if assigning 'color' to 'edge' is valid."""
        for cycle in cycles_by_edge[edge]:
            other_colors = set()
            for oe in cycle:
                if oe != edge:
                    other_colors.add(colors[oe])
            if len(other_colors) == 3 and color not in other_colors:
                return False
        return True
    
    def count_colors(colors):
        return len(set(colors))
    
    def greedy_init(seed):
        """Greedy initialization with randomization."""
        rng = random.Random(seed)
        colors = [0] * NUM_EDGES
        used_colors = set([0])
        
        # Process edges in order of constraint tightness
        edge_order = sorted(range(NUM_EDGES), key=lambda e: len(cycles_by_edge[e]), reverse=True)
        rng.shuffle(edge_order)
        
        for e in edge_order:
            found = False
            # Try new colors first (in random order)
            available = [c for c in range(1, NUM_EDGES) if c not in used_colors]
            rng.shuffle(available)
            for c in available:
                if is_valid_assignment(colors, e, c):
                    colors[e] = c
                    used_colors.add(c)
                    found = True
                    break
            if not found:
                # Try existing colors
                existing = list(used_colors)
                rng.shuffle(existing)
                for c in existing:
                    if is_valid_assignment(colors, e, c):
                        colors[e] = c
                        found = True
                        break
            if not found:
                colors[e] = 0  # fallback
        
        return colors
    
    def local_search(colors, deadline, seed):
        """Simulated annealing local search."""
        rng = random.Random(seed)
        best_colors = colors[:]
        best_count = count_colors(colors)
        current_count = best_count
        
        # Temperature schedule
        T = 0.5
        cooling = 0.995
        min_T = 0.01
        
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < deadline:
            T = max(T * cooling, min_T)
            
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            cur = colors[e]
            
            # Try a random new color
            nc = rng.randint(0, NUM_EDGES - 1)
            if nc == cur:
                continue
            
            if not is_valid_assignment(colors, e, nc):
                continue
            
            # Compute change in color count
            old_count = current_count
            colors[e] = nc
            new_count = count_colors(colors)
            
            if new_count >= old_count:
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
                stuck_count = 0
            else:
                # Accept with probability exp(-delta/T)
                delta = old_count - new_count
                if rng.random() < math.exp(-delta / T):
                    current_count = new_count
                    stuck_count = 0
                else:
                    colors[e] = cur
                    stuck_count += 1
            
            if stuck_count > max_stuck:
                # Perturb: make several random valid changes
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    pe = rng.randint(0, NUM_EDGES - 1)
                    pc = rng.randint(0, NUM_EDGES - 1)
                    if pc == colors[pe]:
                        continue
                    if is_valid_assignment(colors, pe, pc):
                        colors[pe] = pc
                        current_count = count_colors(colors)
                stuck_count = 0
                T = max(T * 1.5, 0.1)  # reheat
        
        return best_colors, best_count
    
    # Multi-start with different seeds
    best_overall = None
    best_overall_count = 0
    num_restarts = 0
    
    # First: try several greedy initializations
    for seed in range(20):
        if time.time() > deadline:
            break
        colors = greedy_init(seed)
        c = count_colors(colors)
        if c > best_overall_count:
            best_overall_count = c
            best_overall = colors[:]
        num_restarts += 1
    
    # Then: local search on best and a few others
    if best_overall is not None:
        result, rc = local_search(best_overall[:], deadline, 42)
        if rc > best_overall_count:
            best_overall_count = rc
            best_overall = result
    
    # Try local search on other good initializations
    for seed in [1, 5, 10, 15]:
        if time.time() > deadline:
            break
        colors = greedy_init(seed)
        c = count_colors(colors)
        if c >= best_overall_count - 2:  # only search if close to best
            result, rc = local_search(colors, deadline, seed * 100)
            if rc > best_overall_count:
                best_overall_count = rc
                best_overall = result
        num_restarts += 1
    
    # Final: more local search iterations on best
    if best_overall is not None and time.time() < deadline:
        result, rc = local_search(best_overall[:], deadline, 999)
        if rc > best_overall_count:
            best_overall_count = rc
            best_overall = result
    
    return best_overall if best_overall is not None else [0] * NUM_EDGES
