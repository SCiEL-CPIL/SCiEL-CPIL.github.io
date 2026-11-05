"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_COLORS = 120

def _get_idx(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i*(i+1)//2 + j - i - 1

def _check_validity(colors):
    """Check that no 4-cycle has four distinct colors."""
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Three 4-cycles on {a,b,c,d}: (a,b,c,d), (a,b,d,c), (a,c,b,d)
        # Cycle 1: a-b, b-c, c-d, d-a
        c1 = {colors[_get_idx(a,b)], colors[_get_idx(b,c)], colors[_get_idx(c,d)], colors[_get_idx(d,a)]}
        if len(c1) == 4:
            return False
        # Cycle 2: a-b, b-d, d-c, c-a
        c2 = {colors[_get_idx(a,b)], colors[_get_idx(b,d)], colors[_get_idx(d,c)], colors[_get_idx(c,a)]}
        if len(c2) == 4:
            return False
        # Cycle 3: a-c, c-b, b-d, d-a
        c3 = {colors[_get_idx(a,c)], colors[_get_idx(c,b)], colors[_get_idx(b,d)], colors[_get_idx(d,a)]}
        if len(c3) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe
    
    random.seed(42)
    
    # Strategy: Start with a 2-coloring based on vertex partition
    # Split vertices into two groups, color within-group edges with color 0, between-group with color 1
    # Then try to introduce more colors via local search
    
    best_colors = None
    best_count = 0
    
    # Initial: use a structured 2-coloring
    # Partition: first 8 vertices in group A, last 8 in group B
    colors = [0] * (N*(N-1)//2)
    for i in range(N):
        for j in range(i+1, N):
            if i < 8 and j < 8:
                colors[_get_idx(i,j)] = 0
            elif i >= 8 and j >= 8:
                colors[_get_idx(i,j)] = 1
            else:
                colors[_get_idx(i,j)] = 2
    
    if _check_validity(colors):
        best_colors = colors[:]
        best_count = _count_colors(colors)
    
    # Local search: try to change edges to new colors while maintaining validity
    edges = list(range(N*(N-1)//2))
    random.shuffle(edges)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Try each edge in random order
        for e in edges:
            if time.time() > deadline:
                break
            
            old_color = colors[e]
            # Try assigning a new color (prefer unused colors)
            used_colors = set(colors)
            unused = [c for c in range(NUM_COLORS) if c not in used_colors]
            
            if unused:
                # Try to assign an unused color
                new_color = random.choice(unused)
            else:
                # Try to reassign to a different used color
                candidates = [c for c in range(NUM_COLORS) if c != old_color]
                if not candidates:
                    continue
                new_color = random.choice(candidates)
            
            colors[e] = new_color
            if _check_validity(colors):
                cur_count = _count_colors(colors)
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]
            else:
                colors[e] = old_color
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 5:
        # Try systematic introduction of colors
        used = set(best_colors)
        for new_c in range(NUM_COLORS):
            if new_c in used:
                continue
            if time.time() > deadline:
                break
            
            # Try to add new_c to one edge at a time
            for e in edges:
                if time.time() > deadline:
                    break
                old = best_colors[e]
                best_colors[e] = new_c
                if _check_validity(best_colors):
                    best_count = _count_colors(best_colors)
                    used.add(new_c)
                else:
                    best_colors[e] = old
    
    if best_colors is None:
        return [0] * (N*(N-1)//2)
    
    return best_colors
# EVOLVE-BLOCK-END
