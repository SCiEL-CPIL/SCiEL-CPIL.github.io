import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles and edge indices
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    vertices = list(range(N))
    
    # Map edge index to (u, v) for easier manipulation
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_map[idx] = (i, j)
            idx += 1
            
    # Precompute for each edge, the list of cycles it belongs to
    # And for each cycle, the list of edge indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    cycle_edges = []
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)
        cycle_edges.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(len(cycle_edges)-1)
        edge_to_cycles[e2].append(len(cycle_edges)-1)
        edge_to_cycles[e3].append(len(cycle_edges)-1)
        edge_to_cycles[e4].append(len(cycle_edges)-1)
        
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)
        cycle_edges.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(len(cycle_edges)-1)
        edge_to_cycles[e2].append(len(cycle_edges)-1)
        edge_to_cycles[e3].append(len(cycle_edges)-1)
        edge_to_cycles[e4].append(len(cycle_edges)-1)
        
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)
        cycle_edges.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(len(cycle_edges)-1)
        edge_to_cycles[e2].append(len(cycle_edges)-1)
        edge_to_cycles[e3].append(len(cycle_edges)-1)
        edge_to_cycles[e4].append(len(cycle_edges)-1)

    num_cycles = len(cycle_edges)
    cycles_arr = np.array(cycle_edges, dtype=np.int32) # shape (360, 4)

    # 2. Initial State: Construct a valid base using disjoint triangles
    # We want to maximize colors. 
    # Strategy: 
    # Start with all zeros.
    # Then, try to introduce new colors one by one.
    # A color is "new" if it's not currently used.
    # We can assign a new color to an edge if doing so doesn't create a violation.
    # Since we start with all 0s, any single edge recolor to a new color c != 0 
    # will only create a violation if there is a 4-cycle containing that edge 
    # where the other 3 edges already have 3 distinct colors (none of which is c).
    # But initially all other edges are 0. So if we recolor edge e to c, 
    # any cycle containing e will have colors {c, 0, 0, 0} or {c, 0, 0, x} etc.
    # Wait, if we recolor one edge to c, and others are 0, the cycle has colors {c, 0, 0, 0}. 
    # Distinct colors: {c, 0}. Count = 2. Not a violation.
    # So we can recolor ANY edge to ANY new color and it remains valid?
    # No, because as we add more colors, the "other 3" edges might have different colors.
    
    # Greedy Algorithm:
    # 1. Start with all edges color 0.
    # 2. Maintain a set of used colors.
    # 3. Iterate. In each step, try to introduce a new color (the smallest unused integer).
    # 4. To introduce a new color, pick a random edge that is currently color 0 (or any color).
    #    Actually, it's better to pick an edge and try to recolor it to a new color.
    #    If valid, accept. If not, try another edge.
    # 5. To maximize diversity, we should try to spread colors out.
    
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_colors = current_colors.copy()
    best_score = 1
    
    def evaluate(colors):
        """
        Returns (num_unique_colors, num_violations)
        """
        num_unique = len(np.unique(colors))
        
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        num_violations = np.sum(distinct_mask)
        
        return num_unique, num_violations

    # Initial evaluation
    curr_score, curr_violations = evaluate(current_colors)
    
    rng = random.Random(42)
    
    # Greedy Phase: Try to introduce new colors
    # We will try to add colors 1, 2, 3... up to MAX_COLORS-1
    # For each new color, we try to find an edge to recolor to this new color.
    # We prioritize edges that are "least constrained" or just random.
    
    used_colors = set([0])
    
    # To make the search more effective, we can use a local search to "make room" for new colors
    # or just greedily assign if possible.
    
    # Let's combine Greedy Introduction with Local Search Refinement.
    
    # Phase 1: Greedy Introduction
    # We will attempt to introduce each new color.
    # For each new color c, we try random edges until we find one that can take color c.
    # If we fail for a color, we might skip it or try later.
    
    # Actually, a simpler robust strategy:
    # 1. Start with all 0s.
    # 2. Randomly pick an edge and a new color.
    # 3. If valid, keep it.
    # 4. Repeat until no more new colors can be added or time limit.
    # This might get stuck.
    
    # Better: 
    # 1. Start with all 0s.
    # 2. While time < deadline:
    #    a. Pick a random edge.
    #    b. Pick a random color (prefer unused ones).
    #    c. If valid, accept.
    #    d. If invalid, maybe try to fix by recoloring another edge? (Too complex)
    #    e. Just reject.
    
    # This is basically random hill climbing.
    
    # Let's use the "Disjoint Triangles" idea from the knowledge.
    # "Using a union of disjoint triangles for unique colors... guarantees no rainbow 4-cycles while using 16 distinct colors"
    # Can we do better?
    # A triangle (u,v,w) has edges (u,v), (v,w), (w,u).
    # If we color a triangle with 3 distinct colors, say 1, 2, 3.
    # Any 4-cycle involving these edges must share at least one edge with the triangle?
    # No. A 4-cycle is on 4 vertices.
    # If the 4 vertices are u, v, w, x.
    # Edges in K4: uv, vw, wu, ux, vx, wx.
    # If uv, vw, wu are colored 1, 2, 3.
    # The 4-cycles are:
    # 1. u-v-x-w-u: edges uv(1), vx(?), xw(?), wu(3).
    # 2. u-v-w-x-u: edges uv(1), vw(2), wx(?), xu(?).
    # 3. u-w-v-x-u: edges uw(3), wv(2), vx(?), xu(?).
    
    # If we set all other edges to 0.
    # Cycle 1: 1, 0, 0, 3 -> {1,0,3}. OK.
    # Cycle 2: 1, 2, 0, 0 -> {1,2,0}. OK.
    # Cycle 3: 3, 2, 0, 0 -> {3,2,0}. OK.
    
    # So if we partition K16 into disjoint triangles and color each triangle with 3 unique colors,
    # and all other edges 0, is it valid?
    # Consider a 4-cycle that uses edges from TWO different triangles.
    # Triangle 1: u,v,w. Triangle 2: x,y,z.
    # 4-cycle u-v-x-y-u? Edges uv, vx, xy, yu.
    # uv is in Tri 1 (color 1). xy is in Tri 2 (color 4).
    # vx and yu are "cross" edges (color 0).
    # Colors: 1, 0, 4, 0. Distinct: {1,0,4}. OK.
    
    # What if the 4-cycle uses edges from 4 different triangles?
    # u in T1, v in T2, w in T3, x in T4.
    # Edges uv, vw, wx, xu. All are cross edges (0).
    # Colors: 0,0,0,0. OK.
    
    # What if it uses 2 edges from T1, 1 from T2, 1 from T3?
    # Vertices: u,v in T1. w in T2. x in T3.
    # Cycle u-v-w-x-u.
    # Edges: uv (in T1, color 1), vw (cross, 0), wx (cross, 0), xu (cross, 0).
    # Colors: 1,0,0,0. OK.
    
    # It seems that if we color disjoint triangles with distinct colors and everything else 0,
    # we are safe.
    # Number of colors: 3 * (number of triangles).
    # 16 vertices -> 5 triangles (15 vertices) + 1 vertex.
    # 5 * 3 = 15 colors + 0 = 16 colors.
    # This matches the knowledge: "16 distinct colors".
    
    # Can we do better?
    # The seed got 15. The inspirations got 16.
    # Can we get 17, 18, ...?
    
    # The constraint is "no 4-cycle with 4 distinct colors".
    # This is equivalent to saying that the set of colors on any 4-cycle has size <= 3.
    
    # Let's try to improve on the 16-color base.
    # We can try to recolor some "0" edges to new colors, or recolor edges within triangles.
    
    # Strategy:
    # 1. Build the 16-color base.
    # 2. Use Local Search to try to increase the number of colors.
    #    - Pick a random edge with color 0.
    #    - Try to recolor it to a new color (e.g., 16, 17, ...).
    #    - If valid, accept.
    #    - If invalid, maybe recolor another edge to "fix" it?
    
    # Let's implement the base construction first.
    
    # Partition vertices into triangles
    triangles = []
    remaining = list(range(N))
    while len(remaining) >= 3:
        t = (remaining.pop(), remaining.pop(), remaining.pop())
        triangles.append(t)
        
    # Color the triangles
    color_id = 1
    for u, v, w in triangles:
        e_uv = get_idx(u, v)
        e_vw = get_idx(v, w)
        e_wu = get_idx(w, u)
        current_colors[e_uv] = color_id
        current_colors[e_vw] = color_id + 1
        current_colors[e_wu] = color_id + 2
        color_id += 3
        
    # Update best
    curr_score, curr_violations = evaluate(current_colors)
    if curr_violations == 0:
        best_score = curr_score
        best_colors = current_colors.copy()
    else:
        # Should not happen
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_colors = current_colors.copy()
        best_score = 1
        curr_score = 1
        curr_violations = 0

    # Local Search to improve
    # We will try to introduce new colors or replace existing ones to increase diversity.
    # Objective: Maximize number of unique colors.
    
    T = 50.0
    cooling = 0.999
    T_min = 0.1
    
    steps = 0
    while time.time() < deadline:
        # Generate a move
        # 1. Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        
        # 2. Pick a new color
        # Bias towards unused colors to increase diversity
        if rng.random() < 0.5:
            # Try an unused color
            used = set(current_colors)
            unused = [c for c in range(MAX_COLORS) if c not in used]
            if unused:
                new_c = rng.choice(unused)
            else:
                new_c = rng.randint(0, MAX_COLORS - 1)
        else:
            new_c = rng.randint(0, MAX_COLORS - 1)
            
        old_c = current_colors[e]
        
        if new_c == old_c:
            continue
            
        current_colors[e] = new_c
        
        new_score, new_violations = evaluate(current_colors)
        
        # Acceptance
        # If valid and score > best_score, update best.
        if new_violations == 0 and new_score > best_score:
            best_score = new_score
            best_colors = current_colors.copy()
            
        # SA Acceptance
        # Fitness: score - penalty * violations
        # We want to explore invalid states if they lead to higher scores later?
        # But we only keep valid bests.
        # Let's use a fitness that strongly penalizes violations.
        old_fitness = curr_score - 100.0 * curr_violations
        new_fitness = new_score - 100.0 * new_violations
        
        delta = new_fitness - old_fitness
        
        if delta >= 0:
            # Accept
            pass
        else:
            if rng.random() < np.exp(delta / T):
                # Accept
                pass
            else:
                # Reject
                current_colors[e] = old_c
                T = max(T_min, T * cooling)
                steps += 1
                continue
        
        curr_score = new_score
        curr_violations = new_violations
        T = max(T_min, T * cooling)
        steps += 1
        
        # Optional: If we have a valid state with high score, try to "polish"
        # by trying to introduce one more color.
        
    # Final Verification
    final_score, final_violations = evaluate(best_colors)
    if final_violations != 0:
        # Fallback to a known valid state (all zeros or the triangle base)
        # Reconstruct triangle base as fallback
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        color_id = 1
        remaining = list(range(N))
        while len(remaining) >= 3:
            t = (remaining.pop(), remaining.pop(), remaining.pop())
            u, v, w = t
            e_uv = get_idx(u, v)
            e_vw = get_idx(v, w)
            e_wu = get_idx(w, u)
            current_colors[e_uv] = color_id
            current_colors[e_vw] = color_id + 1
            current_colors[e_wu] = color_id + 2
            color_id += 3
        return current_colors.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
