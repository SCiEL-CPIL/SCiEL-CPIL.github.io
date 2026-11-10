import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute cycles as numpy array for vectorized operations
CYCLES_ARR = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _count_violations(colors):
    """Vectorized: count 4-cycles with all 4 distinct colors."""
    # colors is a numpy array of shape (120,)
    c = colors[CYCLES_ARR]  # shape (NUM_CYCLES, 4)
    # A cycle is a violation if all 4 colors are distinct
    # c[:,0] != c[:,1] and c[:,0] != c[:,2] and c[:,0] != c[:,3] and c[:,1] != c[:,2] and c[:,1] != c[:,3] and c[:,2] != c[:,3]
    v = ((c[:,0] != c[:,1]) & (c[:,0] != c[:,2]) & (c[:,0] != c[:,3]) &
         (c[:,1] != c[:,2]) & (c[:,1] != c[:,3]) & (c[:,2] != c[:,3]))
    return int(np.sum(v))

def _make_structured_baseline():
    """
    Create a structured coloring based on partitioning vertices into groups.
    Color edge (i,j) based on which groups i and j belong to.
    This gives a valid coloring (no rainbow C4) with moderate diversity.
    
    Strategy: Partition 16 vertices into 4 groups of 4.
    Edges within a group get one color, edges between groups get colors based on group pair.
    Within-group: 4 groups * 6 edges = 24 edges, 1 color each (4 colors)
    Between-group: 6 pairs of groups * 16 edges = 96 edges, 1 color per pair (6 colors)
    Total: 10 colors, all valid since any C4 either has all 4 in same group (1 color),
    or spans at most 3 groups (at most 3 distinct inter-group colors + 1 intra = still <4 distinct
    in a way that avoids rainbow).
    
    Actually let me think more carefully. A 4-cycle on vertices from groups...
    If all 4 vertices are in the same group: all 4 edges are intra-group, 1 color. OK.
    If 3 in one group, 1 in another: edges within the 3-group are intra (color A), 
    edges from the singleton to the group are inter (color B). Cycle uses 2 edges of type A and 2 of type B. 2 colors. OK.
    If 2 in group X, 2 in group Y: edges within X (color A), within Y (color B), between X and Y (color C).
    A 4-cycle would use 2 intra-X, 2 intra-Y? No. Let's see: vertices x1,x2 in X, y1,y2 in Y.
    Cycle x1-x2-y1-y2-x1: edges x1x2 (A), x2y1 (C), y1y2 (B), y2x1 (C). Colors: A,B,C. 3 colors. OK.
    Cycle x1-y1-x2-y2-x1: edges x1y1 (C), y1x2 (C), x2y2 (C), y2x1 (C). All C. 1 color. OK.
    Cycle x1-x2-y2-y1-x1: edges x1x2 (A), x2y2 (C), y2y1 (B), y1x1 (C). Colors: A,B,C. 3 colors. OK.
    If 2 in X, 1 in Y, 1 in Z: edges within X (A), X-Y (B), X-Z (C), Y-Z (D).
    Cycle x1-x2-y1-z1-x1: edges x1x2(A), x2y1(B), y1z1(D), z1x1(C). 4 colors! VIOLATION!
    
    So the simple group-based approach doesn't work for 4 groups. Let me use 2 groups of 8.
    
    2 groups of 8: intra-group color 0, inter-group color 1. Only 2 colors. Too few.
    
    Let me try: partition into 4 groups of 4, but use a more careful coloring.
    For inter-group edges between group p and group q (p<q), use color p*4+q or similar.
    But the problem is the 2-1-1 case.
    
    Alternative: use a 2-coloring structure. Color based on parity of vertex index.
    Even vertices: 0,2,4,...,14 (8 vertices)
    Odd vertices: 1,3,5,...,15 (8 vertices)
    Edges within even: color 0
    Edges within odd: color 1
    Edges between even and odd: color 2
    Only 3 colors.
    
    Better: use a modular structure. Color edge (i,j) = (i+j) mod k for some k.
    This is a well-known construction. For k=4:
    Colors are 0,1,2,3.
    Any 4-cycle a-b-c-d-a: colors are (a+b)%4, (b+c)%4, (c+d)%4, (d+a)%4.
    Sum of these = 2(a+b+c+d) mod 4. If the sum is 0 mod 4, all could be same.
    But can we get 4 distinct? (a+b)%4, (b+c)%4, (c+d)%4, (d+a)%4.
    If a=0,b=1,c=2,d=3: colors 1,3,1,3. Only 2 colors.
    If a=0,b=0,c=1,d=1: colors 0,1,2,1. 3 colors.
    Actually for mod 4, can we get 4 distinct? We need (a+b)%4, (b+c)%4, (c+d)%4, (d+a)%4 all different.
    Let x=(a+b)%4, y=(b+c)%4, z=(c+d)%4, w=(d+a)%4.
    x+w = (a+b+d+a)%4 = (2a+b+d)%4
    y+z = (b+c+c+d)%4 = (b+2c+d)%4
    x+y+z+w = 2(a+b+c+d)%4
    For 4 distinct values in {0,1,2,3}, sum = 6. So 2(a+b+c+d)%4 = 6%4 = 2.
    So a+b+c+d ≡ 1 (mod 2), i.e., odd sum.
    Example: a=0,b=0,c=0,d=1: sum=1. Colors: 0,0,1,1. Not distinct.
    a=0,b=1,c=0,d=0: sum=1. Colors: 1,1,0,0. Not distinct.
    a=0,b=1,c=1,d=1: sum=3. Colors: 1,2,2,1. Not distinct.
    a=0,b=0,c=1,d=1: sum=2 (even). Colors: 0,1,2,1. 3 distinct.
    
    It seems mod 4 might not produce rainbow C4s. Let me verify with a brute force check in my head...
    Actually, this is a known result: edge coloring of complete graph by (i+j) mod k avoids rainbow C4s for certain k.
    
    Let me just use mod 4 as baseline (4 colors) and then try to extend.
    """
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i+1, N):
            colors[_edge_index(i,j)] = (i + j) % 4
    return colors

def _try_batch_assignment(colors, edges_to_try, new_colors):
    """
    Try to assign new_colors to edges_to_try simultaneously.
    Returns True if the assignment is valid (no new violations), and applies it if so.
    """
    n_edges = len(edges_to_try)
    if n_edges == 0:
        return False
    
    # Save old colors
    old_colors = colors[edges_to_try].copy()
    
    # Apply new colors
    colors[edges_to_try] = new_colors
    
    # Check violations
    v = _count_violations(colors)
    
    if v == 0:
        return True
    else:
        # Revert
        colors[edges_to_try] = old_colors
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    random.seed(42)
    
    # Start with structured baseline
    colors = _make_structured_baseline()
    best_colors = colors.copy()
    best_distinct = len(set(colors.tolist()))
    
    # Verify baseline is valid
    assert _count_violations(colors) == 0, "Baseline should be valid"
    
    # Now try to increase diversity
    # Strategy: try to introduce new colors by modifying edges in batches
    # For each edge, determine which colors are "safe" to assign
    
    # Precompute: for each edge, the cycles it belongs to
    # When we change edge e to color c, it's safe if no cycle containing e becomes a violation
    
    # Vectorized approach: for a candidate assignment, just check all cycles at once
    
    # Phase 1: Try to introduce new colors one at a time on individual edges
    # For efficiency, try multiple edges at once
    
    used_colors = set(colors.tolist())
    
    # For each edge, precompute the "constraint": what colors would make it a violation
    # Actually, let's just use trial and error with vectorized checking
    
    # Phase 1: Greedy - try to add new colors
    # Pick random edges, try random new colors, check if valid
    
    for iteration in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Try to introduce a new color on a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try a new color (not currently used, or any different color)
        if len(used_colors) < MAX_COLOR + 1:
            # Try to introduce a genuinely new color
            # Pick from unused colors
            unused_candidates = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused_candidates:
                # Try up to 3 random unused colors
                for _ in range(3):
                    new_color = random.choice(unused_candidates)
                    if new_color == old_color:
                        continue
                    # Try the change
                    colors[e] = new_color
                    if _count_violations(colors) == 0:
                        used_colors.add(new_color)
                        break
                    else:
                        colors[e] = old_color
                else:
                    # No new color worked, try a used color
                    other_used = [c for c in used_colors if c != old_color]
                    if other_used:
                        new_color = random.choice(other_used)
                        colors[e] = new_color
                        if _count_violations(colors) == 0:
                            pass
                        else:
                            colors[e] = old_color
            else:
                # All colors used
                new_color = random.randint(0, MAX_COLOR)
                if new_color != old_color:
                    colors[e] = new_color
                    if _count_violations(colors) == 0:
                        pass
                    else:
                        colors[e] = old_color
        else:
            # All colors used, just shuffle
            new_color = random.randint(0, MAX_COLOR)
            if new_color != old_color:
                colors[e] = new_color
                if _count_violations(colors) == 0:
                    pass
                else:
                    colors[e] = old_color
        
        # Update used colors
        current_distinct = len(set(colors.tolist()))
        if current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = colors.copy()
    
    # Phase 2: Try batch assignments for more aggressive exploration
    # Pick a random set of edges and try to assign them new colors simultaneously
    
    for iteration in range(20000):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random batch of edges
        batch_size = random.randint(1, 5)
        edges_batch = random.sample(range(NUM_EDGES), batch_size)
        
        # For each edge in the batch, pick a new color
        old_vals = colors[edges_batch].copy()
        new_vals = np.array([random.randint(0, MAX_COLOR) for _ in range(batch_size)], dtype=np.int32)
        
        # Make sure we're actually changing something
        if np.array_equal(old_vals, new_vals):
            continue
        
        colors[edges_batch] = new_vals
        if _count_violations(colors) == 0:
            current_distinct = len(set(colors.tolist()))
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors.copy()
        else:
            colors[edges_batch] = old_vals
    
    # Final verification
    final_colors = best_colors.tolist()
    if _count_violations(np.array(final_colors, dtype=np.int32)) != 0:
        # Fallback to baseline
        return _make_structured_baseline().tolist()
    
    return final_colors
