import time
from itertools import combinations
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def idx_to_edge(idx):
    """Convert lexicographic edge index back to (i, j)."""
    # Binary search or direct computation
    # Row i starts at index: i*(N-1) - i*(i-1)//2
    # For a given idx, find i such that start_i <= idx < start_{i+1}
    lo, hi = 0, N - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        start_mid = mid * (N - 1) - mid * (mid - 1) // 2
        if start_mid <= idx:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    start_i = i * (N - 1) - i * (i - 1) // 2
    pos = idx - start_i
    j = i + 1 + pos
    return i, j

def get_4cycles():
    """Precompute all 4-cycles (3 per 4-vertex subset)."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three possible 4-cycles on {a,b,c,d}:
        # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
        # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
        # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        cycles.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    return cycles

def build_gallai_initial():
    """Build the initial Gallai-decomposition coloring with improved DP."""
    dp = [0] * (N + 1)
    split = [0] * (N + 1)
    dp[1] = 0
    dp[2] = 1
    dp[3] = 3
    dp[4] = 4
    for n in range(5, N + 1):
        best_val = 0
        best_k = 1
        for k in range(1, n):
            val = dp[k] + dp[n - k] + 1
            if val > best_val:
                best_val = val
                best_k = k
        dp[n] = best_val
        split[n] = best_k
        
    colors = [0] * NUM_EDGES
    
    def assign_edge(u, v, c):
        colors[edge_idx(u, v)] = c
        
    def build_block(vertices, start_color):
        n = len(vertices)
        if n == 1:
            return start_color
        if n == 2:
            u, v = vertices
            assign_edge(u, v, start_color)
            return start_color + 1
        if n == 3:
            u, v, w = vertices
            assign_edge(u, v, start_color)
            assign_edge(u, w, start_color + 1)
            assign_edge(v, w, start_color + 2)
            return start_color + 3
        if n == 4:
            u, v, w, x = vertices
            A = start_color
            B = start_color + 1
            C = start_color + 2
            D = start_color + 3
            assign_edge(u, v, A)
            assign_edge(w, x, A)
            assign_edge(u, w, B)
            assign_edge(v, x, B)
            assign_edge(u, x, C)
            assign_edge(v, w, D)
            return start_color + 4
        k = split[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        next_color = build_block(v1, start_color)
        next_color = build_block(v2, next_color)
        cross_color = next_color
        for u in v1:
            for v in v2:
                assign_edge(u, v, cross_color)
        return cross_color + 1
        
    build_block(list(range(N)), 0)
    return colors

def build_modular_initial(a, b, mod):
    """Build a modular arithmetic coloring: c(i,j) = (a*i + b*j) % mod."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = (a * i + b * j) % mod
    return colors

def build_mixed_initial():
    """Build a mixed initial coloring using multiple modular patterns."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_idx(i, j)
            # Use different modular formulas for different edge groups
            if (i + j) % 3 == 0:
                colors[idx] = (i * j) % 19
            elif (i + j) % 3 == 1:
                colors[idx] = (i + 3 * j) % 17
            else:
                colors[idx] = (2 * i + 5 * j) % 23
    return colors

def check_validity(colors, cycles):
    """Check if no 4-cycle has four distinct colors."""
    for e1, e2, e3, e4 in cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def local_search(initial_colors, cycles, time_limit=55.0):
    """Local search to try to increase color count using multiple strategies."""
    start_time = time.time()
    colors = list(initial_colors)
    best_colors = list(initial_colors)
    best_count = len(set(colors))
    
    # Precompute for each edge, which cycles it participates in
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_cycles[e].append(ci)
    
    # Count how many times each color is used
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    random.seed(42)
    
    def try_recolor(edge, new_color):
        """Check if recoloring edge to new_color maintains validity."""
        for ci in edge_cycles[edge]:
            e1, e2, e3, e4 = cycles[ci]
            c1 = new_color if e1 == edge else colors[e1]
            c2 = new_color if e2 == edge else colors[e2]
            c3 = new_color if e3 == edge else colors[e3]
            c4 = new_color if e4 == edge else colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
    
    def commit_recolor(edge, old_color, new_color):
        """Apply a recolor and update counts."""
        colors[edge] = new_color
        color_counts[old_color] -= 1
        if color_counts[old_color] == 0:
            del color_counts[old_color]
        color_counts[new_color] = color_counts.get(new_color, 0) + 1
    
    # Phase 1: Systematic new color introduction - try all edges in order
    for phase_iter in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Try each edge systematically
        for edge in range(NUM_EDGES):
            if time.time() - start_time > time_limit:
                break
            old_color = colors[edge]
            
            # Try to introduce a brand new color
            used = set(color_counts.keys())
            new_color = 0
            while new_color in used:
                new_color += 1
            if new_color >= 120:
                break
            
            if try_recolor(edge, new_color):
                commit_recolor(edge, old_color, new_color)
                current_count = len(color_counts)
                if current_count > best_count:
                    best_count = current_count
                    best_colors = list(colors)
    
    # Phase 2: Randomized search with focus on new colors
    for phase_iter in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        edge = random.randint(0, NUM_EDGES - 1)
        old_color = colors[edge]
        
        # Try several candidate new colors
        used = set(color_counts.keys())
        # Find smallest unused color
        new_color = 0
        while new_color in used:
            new_color += 1
        if new_color >= 120:
            # Try random existing colors for consolidation
            if color_counts:
                new_color = random.choice(list(color_counts.keys()))
                if new_color == old_color:
                    continue
        else:
            # With some probability, try a random unused color
            if random.random() < 0.3:
                new_color = random.randint(0, 119)
                if new_color in used:
                    new_color = 0
                    while new_color in used:
                        new_color += 1
                    if new_color >= 120:
                        continue
        
        if try_recolor(edge, new_color):
            commit_recolor(edge, old_color, new_color)
            current_count = len(color_counts)
            if current_count > best_count:
                best_count = current_count
                best_colors = list(colors)
    
    # Phase 3: Final polish - systematic search for new colors
    for phase_iter in range(50):
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        for edge in range(NUM_EDGES):
            if time.time() - start_time > time_limit:
                break
            old_color = colors[edge]
            
            # Try to find any new color that works
            used = set(color_counts.keys())
            for new_color in range(120):
                if new_color in used:
                    continue
                if try_recolor(edge, new_color):
                    commit_recolor(edge, old_color, new_color)
                    progress = True
                    current_count = len(color_counts)
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = list(colors)
                    break
        
        if not progress:
            break
    
    return best_colors, best_count

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Uses multiple initial colorings and local search to maximize colors.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    cycles = get_4cycles()
    
    # Try multiple initial colorings
    initial_colorings = []
    
    # 1. Gallai decomposition
    gallai = build_gallai_initial()
    if check_validity(gallai, cycles):
        initial_colorings.append(gallai)
    
    # 2. Various modular colorings
    modular_params = [
        (1, 1, 19), (1, 2, 19), (1, 3, 17), (2, 1, 19),
        (1, 1, 17), (1, 2, 23), (2, 3, 19), (3, 1, 17),
        (1, 4, 19), (2, 5, 23), (1, 1, 23), (3, 2, 19),
    ]
    for a, b, mod in modular_params:
        mod_colors = build_modular_initial(a, b, mod)
        if check_validity(mod_colors, cycles):
            initial_colorings.append(mod_colors)
    
    # 3. Mixed initial
    mixed = build_mixed_initial()
    if check_validity(mixed, cycles):
        initial_colorings.append(mixed)
    
    # If no valid initial coloring found, use all-same color
    if not initial_colorings:
        return [0] * NUM_EDGES
    
    # Run local search on each initial coloring, keep best
    best_colors = None
    best_count = 0
    
    # Give more time to the Gallai initial (index 0) since it has the best base
    time_weights = [3.0] + [1.0] * (len(initial_colorings) - 1)
    total_weight = sum(time_weights)
    remaining_time = deadline - (time.time() - start_time)
    
    for i, init in enumerate(initial_colorings):
        if time.time() - start_time > deadline - 2.0:
            break
        
        time_share = (time_weights[i] / total_weight) * remaining_time
        local_time = min(time_share, deadline - (time.time() - start_time) - 2.0)
        if local_time < 1.0:
            local_time = 1.0
        
        colors, count = local_search(init, cycles, time_limit=local_time)
        
        # Verify result
        if check_validity(colors, cycles):
            if count > best_count:
                best_count = count
                best_colors = colors
        else:
            # Fall back to initial if local search failed
            if count > best_count:
                best_count = count
                best_colors = init
    
    if best_colors is None:
        best_colors = initial_colorings[0]
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
