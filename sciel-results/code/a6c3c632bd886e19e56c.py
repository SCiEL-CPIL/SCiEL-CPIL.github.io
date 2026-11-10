import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))

def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations

def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Strategy: XOR-based structural coloring c(i,j) = h(i XOR j)
    # There are 15 distinct non-zero XOR values for 4-bit indices.
    # For any 4-cycle a-b-c-d, the edge XORs are x1=i1^i2, x2=i2^i3, x3=i3^i4, x4=i4^i1.
    # We have x1 ^ x2 ^ x3 ^ x4 = 0.
    # This imposes a constraint on h: h(x1), h(x2), h(x3), h(x4) cannot be all distinct.
    # Specifically, for any x, y, z with x^y^z != 0, we must have h(x), h(y), h(z), h(x^y^z) not all distinct.
    # If we map all XOR values to a small set of colors, we can structurally avoid rainbow cycles.
    
    # Let's try to find a mapping h: {1..15} -> {0..K-1} that maximizes distinct colors used,
    # subject to the constraint that for all x,y,z in {1..15} with x!=y!=z!=x^y^z,
    # the set {h(x), h(y), h(z), h(x^y^z)} has size <= 3.
    
    # Actually, the constraint is that for ANY 4-cycle, the 4 colors are not distinct.
    # The 4 XOR values in a cycle are x, y, z, w with x^y^z^w=0.
    # So w = x^y^z.
    # The constraint is: for all x,y,z in {1..15} such that w=x^y^z is also in {1..15} (and distinct from x,y,z? No, edges must be distinct, so x,y,z,w must be distinct non-zero values),
    # the colors h(x), h(y), h(z), h(w) are not all distinct.
    
    # Note: If x^y^z = 0, then w=0 which is not an edge. But x,y,z are distinct non-zero, so x^y^z != 0 unless x=y^z which implies x^y=z, so x,y,z not distinct? No.
    # If x,y,z are distinct non-zero 4-bit vectors, x^y^z can be 0 only if x^y=z, i.e., x,y,z are linearly dependent over GF(2).
    # If x^y^z = 0, then the "cycle" would have w=0, which means two vertices are the same, so it's not a simple 4-cycle.
    # So we only care about x,y,z distinct non-zero such that w=x^y^z is also non-zero and distinct from x,y,z.
    # w != 0 means x,y,z not linearly dependent.
    # w != x means y^z != 0 (true since y!=z).
    # w != y means x^z != 0 (true).
    # w != z means x^y != 0 (true).
    # So the condition is simply: x,y,z distinct non-zero 4-bit vectors, and x^y^z != 0.
    # Then we require |{h(x), h(y), h(z), h(x^y^z)}| <= 3.
    
    # Let's precompute the constraints on h.
    # h is a function from 1..15 to 0..119.
    # We want to maximize the number of distinct values in the image of h.
    # But h is only 15 values. So max distinct colors is 15.
    # Wait, the task asks for colors in 0..119 for 120 edges.
    # If we use the XOR strategy, we only use at most 15 colors.
    # The current score is 17. So this strategy might be worse.
    # However, the prompt says "Maximize the number of colors actually used".
    # If the XOR strategy is limited to 15 colors, it's not better than 17.
    
    # Is there a way to use more colors?
    # Maybe h maps XOR values to colors, but we can also permute vertices?
    # No, the vertices are fixed 0..15.
    # The XOR structure is rigid.
    
    # Let's check if we can relax the XOR structure.
    # Maybe c(i,j) = h(i XOR j) is too restrictive.
    # What if we use c(i,j) = h(i) XOR h(j)?
    # Then for a cycle a-b-c-d, colors are h(a)^h(b), h(b)^h(c), h(c)^h(d), h(d)^h(a).
    # Sum (XOR) is 0.
    # Same constraint.
    
    # What if we don't use a group structure?
    # The prompt suggests: "Model the edge coloring as a function of the bitwise XOR of vertex indices... Optimize t"
    # It seems to imply that this strategy is worth trying.
    # But 15 < 17.
    # Unless... the constraint allows more than 15 colors if we break the XOR symmetry?
    # No, if c(i,j) depends ONLY on i^j, then there are only 15 possible values.
    
    # Perhaps the strategy is to start with XOR, then break symmetry with local search?
    # The XOR coloring is a valid starting point (if we find a valid h).
    # Then we can recolor edges to introduce more colors, using the local search from the seed.
    # This combines structural initialization with local optimization.
    
    # Step 1: Find a valid h: {1..15} -> {0..119} maximizing distinct colors.
    # Since max is 15, let's just try to find ANY valid h with 15 distinct colors.
    # If none exists, try 14, 13, etc.
    
    # Constraints: For all x,y,z in 1..15 distinct, x^y^z != 0:
    # h(x), h(y), h(z), h(x^y^z) not all distinct.
    
    # Let's generate all such constraints.
    # There are C(15,3) = 455 triples.
    # For each, check if x^y^z is non-zero and distinct from x,y,z.
    # If so, add constraint.
    
    constraints = []
    for x in range(1, 16):
        for y in range(1, 16):
            if y == x: continue
            for z in range(1, 16):
                if z == x or z == y: continue
                w = x ^ y ^ z
                if w == 0: continue
                if w == x or w == y or w == z: continue
                # Add constraint: h(x), h(y), h(z), h(w) not all distinct
                # To avoid duplicates, sort (x,y,z,w)
                quad = tuple(sorted((x, y, z, w)))
                constraints.append(quad)
    
    # Remove duplicates
    constraints = list(set(constraints))
    # print(len(constraints))
    
    # Now, search for h.
    # Since 15 variables, we can use backtracking or just random search.
    # Let's try random assignment of 15 distinct colors and check validity.
    # If valid, we have a base coloring with 15 colors.
    
    rng = random.Random(123)
    base_h = None
    for _ in range(10000):
        # Assign random distinct colors to 1..15
        colors_for_h = rng.sample(range(120), 15)
        h = {}
        for i in range(15):
            h[i+1] = colors_for_h[i]
        
        valid = True
        for x, y, z, w in constraints:
            c1, c2, c3, c4 = h[x], h[y], h[z], h[w]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                valid = False
                break
        if valid:
            base_h = h
            break
    
    if base_h is None:
        # Fallback: all 0s
        base_h = {i: 0 for i in range(1, 16)}
    
    # Construct initial colors from base_h
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        xor_val = i ^ j
        colors[idx] = base_h[xor_val]
    
    # Now, use local search to increase distinct colors beyond 15.
    # This is where the "Optimize t" part comes in.
    # We start from a valid state with 15 colors and try to add more.
    
    color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    current_distinct = np.sum(color_counts > 0)
    
    best_colors = list(colors)
    best_score = current_distinct
    
    T = 1.0
    T_min = 1e-5
    cooling = 0.999
    
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        if steps % 10000 == 0:
            T *= cooling
            if T < T_min:
                T = T_min
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Recolor
            e = rng.randint(0, NUM_EDGES - 1)
            # Bias towards unused colors
            unused_colors = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
            if unused_colors and rng.random() < 0.6:
                c = rng.choice(unused_colors)
            else:
                c = rng.randint(0, MAX_COLOR - 1)
            
            if check_edge_change(colors, e, c):
                old_c = colors[e]
                if old_c == c:
                    continue
                colors[e] = c
                
                color_counts[old_c] -= 1
                color_counts[c] += 1
                
                if color_counts[old_c] == 0:
                    current_distinct -= 1
                if color_counts[c] == 1:
                    current_distinct += 1
                
                new_score = current_distinct
                delta = new_score - best_score
                
                if delta > 0:
                    best_score = new_score
                    best_colors = list(colors)
                elif delta < 0 and rng.random() < np.exp(delta / T):
                    pass # Accept, but don't update best
                else:
                    # Reject
                    colors[e] = old_c
                    color_counts[c] -= 1
                    color_counts[old_c] += 1
                    if color_counts[c] == 0:
                        current_distinct -= 1
                    if color_counts[old_c] == 1:
                        current_distinct += 1
        else:
            # Swap
            e1 = rng.randint(0, NUM_EDGES - 1)
            e2 = rng.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            if c1 == c2:
                continue
            
            # Check swap validity
            valid = True
            # Check cycles involving e1
            for o1, o2, o3 in OTHER_EDGES[e1]:
                nc1 = c2
                nc_o1 = c1 if o1 == e2 else colors[o1]
                nc_o2 = c1 if o2 == e2 else colors[o2]
                nc_o3 = c1 if o3 == e2 else colors[o3]
                if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check cycles involving e2
            for o1, o2, o3 in OTHER_EDGES[e2]:
                nc1 = c1
                nc_o1 = c2 if o1 == e1 else colors[o1]
                nc_o2 = c2 if o2 == e1 else colors[o2]
                nc_o3 = c2 if o3 == e1 else colors[o3]
                if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
                    valid = False
                    break
            
            if valid:
                colors[e1] = c2
                colors[e2] = c1
                # Distinct count doesn't change with swap
                if current_distinct > best_score:
                    best_score = current_distinct
                    best_colors = list(colors)
        
        # Restart logic
        if steps % 50000 == 0 and T < 0.1:
            T = 0.5
    
    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
    
    return list(best_colors)
