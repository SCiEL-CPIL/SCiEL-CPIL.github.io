import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = MAX_COLOR + 1  # 120

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()

def _validate(colors):
    """Full validation of a coloring."""
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _generate_additive_coloring(p, q, offset, scale):
    """
    Generate colors from additive model: color(i,j) = (p[i] + q[j]) * scale + offset.
    We need to map to 0..119.
    To maximize distinct colors, we want the sums p[i]+q[j] to be as distinct as possible.
    Since p and q are small integers, the sums will cluster.
    However, the structure guarantees no rainbow 4-cycles.
    """
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_index(i, j)
            val = (p[i] + q[j]) * scale + offset
            # Clamp to 0..119
            if val < 0:
                val = 0
            elif val > 119:
                val = 119
            colors[idx] = val
    return colors

def _local_search_additive(p, q, scale, offset, rng, deadline):
    """
    Optimize p, q, scale, offset to maximize distinct colors.
    """
    best_p = p.copy()
    best_q = q.copy()
    best_scale = scale
    best_offset = offset
    best_colors = _generate_additive_coloring(best_p, best_q, best_scale, best_offset)
    best_distinct = _count_distinct(best_colors)
    
    current_p = p.copy()
    current_q = q.copy()
    current_scale = scale
    current_offset = offset
    
    while time.time() < deadline:
        # Randomly choose a move type
        move_type = rng.randint(0, 3)
        
        if move_type == 0:
            # Change a p value
            idx = rng.randint(0, N - 1)
            old_val = current_p[idx]
            current_p[idx] = old_val + rng.randint(-3, 3)
            # Keep p values non-negative and small to avoid overflow/clamping issues
            if current_p[idx] < 0: current_p[idx] = 0
            if current_p[idx] > 10: current_p[idx] = 10
            
        elif move_type == 1:
            # Change a q value
            idx = rng.randint(0, N - 1)
            old_val = current_q[idx]
            current_q[idx] = old_val + rng.randint(-3, 3)
            if current_q[idx] < 0: current_q[idx] = 0
            if current_q[idx] > 10: current_q[idx] = 10
            
        elif move_type == 2:
            # Change scale
            current_scale = max(1, current_scale + rng.randint(-2, 2))
            
        else:
            # Change offset
            current_offset = max(0, min(119, current_offset + rng.randint(-5, 5)))
            
        new_colors = _generate_additive_coloring(current_p, current_q, current_scale, current_offset)
        new_distinct = _count_distinct(new_colors)
        
        # Accept if better or with some probability (simulated annealing style)
        if new_distinct > best_distinct:
            best_p = current_p.copy()
            best_q = current_q.copy()
            best_scale = current_scale
            best_offset = current_offset
            best_colors = new_colors
            best_distinct = new_distinct
        else:
            # Revert if not better (simple hill climbing)
            if move_type == 0:
                current_p[idx] = old_val
            elif move_type == 1:
                current_q[idx] = old_val
            # scale and offset are just ints, so they stay changed unless we revert them?
            # For simplicity, let's just revert everything if not better? No, that's too strict.
            # Let's just accept if it's not worse, to explore flat regions?
            # Actually, let's stick to strict improvement for now to ensure progress, 
            # but add random restarts if stuck.
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_p, best_q, best_scale, best_offset, best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard kill

    best_overall_colors = None
    best_overall_distinct = 0
    rng = random.Random(42)

    # Initial parameters
    # p and q are vertex labels. 
    # If p and q are permutations of 0..15, sums range 0..30.
    # With scale=4, range 0..120. This might give good spread.
    
    for restart in range(50):
        if time.time() >= deadline:
            break
            
        # Initialize p and q
        # Strategy: Random permutations or random small integers
        # Using permutations ensures more distinct sums
        p = np.random.permutation(N)
        q = np.random.permutation(N)
        
        # Try different scales and offsets
        # Sums range 0..30.
        # If scale=4, range 0..120.
        # If scale=3, range 0..90.
        
        scale = rng.randint(3, 5)
        offset = rng.randint(0, 20)
        
        local_deadline = min(time.time() + 2.0, deadline - 1.0)
        
        bp, bq, bs, bo, bcolors, bdistinct = _local_search_additive(p, q, scale, offset, rng, local_deadline)
        
        if bdistinct > best_overall_distinct:
            best_overall_distinct = bdistinct
            best_overall_colors = bcolors.copy()
            
    # If additive model didn't do well, try a hybrid: 
    # Use additive model as a base, then try to tweak individual edges?
    # But tweaking breaks the structural guarantee.
    # So we must stick to the structural model or verify validity.
    
    # Let's try one more strategy: 
    # Random partition of vertices into groups, assign group colors.
    # This is a special case of additive model where p and q are step functions.
    
    # If we have time, try to refine the best additive solution with a generic local search 
    # that checks validity explicitly. This allows breaking the additive structure slightly 
    # if it increases diversity while maintaining validity.
    
    if best_overall_colors is not None and time.time() < deadline - 5.0:
        # Generic local search on the best additive coloring
        colors = best_overall_colors.copy()
        color_counts = np.zeros(NUM_COLORS, dtype=np.int32)
        for c in colors:
            color_counts[c] += 1
            
        # Precompute edge to cycles map for fast validity check
        edge_to_cycles = [[] for _ in range(NUM_EDGES)]
        for ci, cyc in enumerate(CYCLES):
            for e in cyc:
                edge_to_cycles[e].append(ci)
                
        def _is_move_valid(e, new_color, colors_arr):
            old_color = colors_arr[e]
            if old_color == new_color:
                return True
            for ci in edge_to_cycles[e]:
                cyc = CYCLES[ci]
                nc = [colors_arr[x] for x in cyc]
                pos = cyc.index(e)
                nc[pos] = new_color
                if len(set(nc)) == 4:
                    return False
            return True

        current_distinct = _count_distinct(colors)
        best_distinct = current_distinct
        best_colors = colors.copy()
        
        # Local search
        while time.time() < deadline - 1.0:
            improved = False
            for e in range(NUM_EDGES):
                if time.time() >= deadline - 1.0:
                    break
                old_color = colors[e]
                # Try unused colors first
                unused = [c for c in range(NUM_COLORS) if color_counts[c] == 0]
                if unused:
                    rng.shuffle(unused)
                    candidates = unused[:5]
                else:
                    # Try random colors
                    candidates = [rng.randint(0, MAX_COLOR) for _ in range(5)]
                
                for new_color in candidates:
                    if new_color == old_color:
                        continue
                    if _is_move_valid(e, new_color, colors):
                        # Apply
                        colors[e] = new_color
                        color_counts[old_color] -= 1
                        color_counts[new_color] += 1
                        
                        new_distinct = current_distinct
                        if color_counts[old_color] == 0:
                            new_distinct -= 1
                        if color_counts[new_color] == 1:
                            new_distinct += 1
                        
                        if new_distinct > best_distinct:
                            best_distinct = new_distinct
                            best_colors = colors.copy()
                            improved = True
                            current_distinct = new_distinct
                            break
                        else:
                            # Revert
                            colors[e] = old_color
                            color_counts[old_color] += 1
                            color_counts[new_color] -= 1
                            
            if not improved:
                # Perturbation
                for _ in range(3):
                    e = rng.randint(0, NUM_EDGES - 1)
                    new_c = rng.randint(0, MAX_COLOR)
                    old_c = colors[e]
                    if new_c != old_c and _is_move_valid(e, new_c, colors):
                        colors[e] = new_c
                        color_counts[old_c] -= 1
                        color_counts[new_c] += 1
                        current_distinct = _count_distinct(colors)

        if best_distinct > best_overall_distinct:
            best_overall_distinct = best_distinct
            best_overall_colors = best_colors.copy()

    if best_overall_colors is None:
        best_overall_colors = np.zeros(NUM_EDGES, dtype=np.int32)

    # Final validation
    if not _validate(best_overall_colors):
        return [0] * NUM_EDGES

    return best_overall_colors.tolist()
