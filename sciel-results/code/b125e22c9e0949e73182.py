"""New program: hypergraph coloring with entropy-guided simulated annealing."""
import random
import math
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a tuple of 4 edge indices.
    """
    edge_to_idx = _build_edge_index()
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,c), max(b,c))]
        e3 = edge_to_idx[(min(c,d), max(c,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,d), max(b,d))]
        e3 = edge_to_idx[(min(d,c), max(d,c))]
        e4 = edge_to_idx[(min(c,a), max(c,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(min(a,c), max(a,c))]
        e2 = edge_to_idx[(min(c,b), max(c,b))]
        e3 = edge_to_idx[(min(b,d), max(b,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
    return cycles

def _build_edge_to_cycles():
    """Map each edge index to list of cycle indices it participates in."""
    cycles = _build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return edge_to_cycles

# Precompute
_EDGE_TO_IDX = _build_edge_index()
_CYCLES = _build_cycles()
_EDGE_TO_CYCLES = _build_edge_to_cycles()
NUM_CYCLES = len(_CYCLES)

def _compute_entropy(color_counts, num_distinct):
    """Compute Shannon entropy of color distribution (normalized)."""
    if num_distinct == 0:
        return 0.0
    entropy = 0.0
    for c in range(NUM_COLORS):
        if color_counts[c] > 0:
            p = color_counts[c] / NUM_EDGES
            entropy -= p * math.log(p)
    return entropy / math.log(NUM_COLORS)  # normalized to [0,1]

def _initialize_valid():
    """
    Create a valid coloring with moderate diversity using a structured approach.
    Use a proper edge coloring of K16 based on 1-factorization.
    K16 can be edge-colored with 15 colors (proper edge coloring).
    Then we can merge some colors to create a valid base, then diversify.
    Actually, for our constraint (no 4 distinct colors in any 4-cycle),
    a proper edge coloring with k colors where k < 4 per cycle works.
    
    Let's use a simpler approach: assign colors based on a pattern that
    guarantees at most 3 distinct colors per 4-cycle.
    """
    colors = [0] * NUM_EDGES
    
    # Use a modular construction: color edge (i,j) based on (i+j) mod k
    # For k=4, this gives at most 4 colors, but we need to verify no 4-cycle
    # has all 4 distinct. Let's use k=3 to be safe, then diversify.
    
    # Better: use a 1-factorization approach.
    # K16 vertices 0..15. Pair up: for round r in 0..14, 
    # edge (i, (i+r) mod 15) for i != (15-r) mod 15, plus (15, (15+r) mod 15)
    # This gives 15 perfect matchings, each a proper edge coloring.
    
    # Assign each matching a color from 0..14
    # This is a proper 15-edge-coloring, so every 4-cycle has edges from 
    # at most 4 different matchings. But we need at most 3 distinct colors.
    # A proper edge coloring means adjacent edges have different colors,
    # but a 4-cycle could still have 4 distinct colors.
    
    # Simpler valid initialization: use only 2 colors in a checkerboard-like pattern
    # Color (i,j) = (i + j) % 2
    for i in range(N):
        for j in range(i + 1, N):
            idx = _EDGE_TO_IDX[(i, j)]
            colors[idx] = (i + j) % 2
    
    # Verify validity
    color_counts = [0] * NUM_COLORS
    for c in colors:
        color_counts[c] += 1
    num_distinct = sum(1 for c in color_counts if c > 0)
    
    cycle_color_counts = [[0] * NUM_COLORS for _ in range(NUM_CYCLES)]
    for ci, cycle in enumerate(_CYCLES):
        for ei in cycle:
            cycle_color_counts[ci][colors[ei]] += 1
    
    cycle_distinct = [0] * NUM_CYCLES
    for ci in range(NUM_CYCLES):
        for c in range(NUM_COLORS):
            if cycle_color_counts[ci][c] > 0:
                cycle_distinct[ci] += 1
    
    violations = sum(1 for d in cycle_distinct if d == 4)
    
    return colors, color_counts, num_distinct, violations, cycle_color_counts, cycle_distinct

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 55.0  # leave 5s margin
    
    # Initialize
    colors, color_counts, num_distinct, violations, cycle_color_counts, cycle_distinct = _initialize_valid()
    
    best_colors = colors[:]
    best_distinct = num_distinct
    best_entropy = _compute_entropy(color_counts, num_distinct)
    
    # Simulated annealing parameters
    T_start = 3.0
    T_min = 0.005
    cooling = 0.997
    max_iter = 200000
    
    W_V = 500   # Weight for violations
    W_D = 1.0   # Weight for diversity (num distinct colors)
    W_E = 0.5   # Weight for entropy bonus
    
    # Track best
    no_improve = 0
    improve_threshold = 5000
    
    for iteration in range(max_iter):
        if T_start * (cooling ** iteration) < T_min:
            break
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (cooling ** iteration)
        
        # Pick a random edge
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color with entropy-guided bias
        # Higher probability for colors with lower count (to increase entropy)
        # Use a softmax-like sampling
        r = random.random()
        if r < 0.5:
            # Pick an unused color if available
            unused = [c for c in range(NUM_COLORS) if color_counts[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                # Pick a rare color
                rare = [c for c in range(NUM_COLORS) if color_counts[c] <= 2]
                if rare:
                    new_color = random.choice(rare)
                else:
                    new_color = random.randint(0, NUM_COLORS - 1)
        elif r < 0.8:
            # Pick from colors with count <= 3
            rare = [c for c in range(NUM_COLORS) if color_counts[c] <= 3]
            if rare:
                new_color = random.choice(rare)
            else:
                new_color = random.randint(0, NUM_COLORS - 1)
        else:
            new_color = random.randint(0, NUM_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations using cycle_color_counts
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            old_distinct = cycle_distinct[ci]
            old_count = cycle_color_counts[ci][old_color]
            new_count = cycle_color_counts[ci][new_color]
            
            # After removing old_color:
            if old_count == 1:
                temp_distinct = old_distinct - 1
            else:
                temp_distinct = old_distinct
            
            # After adding new_color:
            if new_count == 0:
                new_distinct = temp_distinct + 1
            else:
                new_distinct = temp_distinct
            
            old_bad = (old_distinct == 4)
            new_bad = (new_distinct == 4)
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Calculate delta distinct
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1
        if color_counts[new_color] == 0:
            delta_distinct += 1
            
        # Calculate delta entropy (approximate)
        # Entropy change when moving one edge from old to new
        # dH = -(p_new' * log(p_new') - p_new * log(p_new)) - (p_old' * log(p_old') - p_old * log(p_old))
        # where p = count / NUM_EDGES
        # For small changes, approximate:
        # If old_count goes from k to k-1: delta_p_old = -1/N
        # If new_count goes from m to m+1: delta_p_new = +1/N
        # This is complex; use a simpler proxy:
        # Reward if we're spreading out (reducing max count, increasing min among used)
        delta_entropy_proxy = 0.0
        if color_counts[old_color] > color_counts[new_color]:
            # Moving from more common to less common color: increases entropy
            delta_entropy_proxy = 1.0
        elif color_counts[old_color] < color_counts[new_color]:
            # Moving from less common to more common: decreases entropy
            delta_entropy_proxy = -1.0
            
        # Calculate delta cost
        delta_cost = delta_violations * W_V - delta_distinct * W_D - delta_entropy_proxy * W_E
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
                
        if accept:
            # Apply the move
            colors[ei] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Update cycle_color_counts and cycle_distinct
            for ci in _EDGE_TO_CYCLES[ei]:
                old_distinct = cycle_distinct[ci]
                
                cycle_color_counts[ci][old_color] -= 1
                if cycle_color_counts[ci][old_color] == 0:
                    temp_distinct = old_distinct - 1
                else:
                    temp_distinct = old_distinct
                
                old_new_count = cycle_color_counts[ci][new_color]
                cycle_color_counts[ci][new_color] += 1
                if old_new_count == 0:
                    new_distinct = temp_distinct + 1
                else:
                    new_distinct = temp_distinct
                
                cycle_distinct[ci] = new_distinct
            
            violations += delta_violations
            num_distinct += delta_distinct
            
            # Track best (only valid solutions)
            if violations == 0:
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
                
                # If stuck for too long, restart with perturbation
                if no_improve > improve_threshold:
                    no_improve = 0
                    # Perturb: randomly change some edges
                    for _ in range(10):
                        pei = random.randint(0, NUM_EDGES - 1)
                        pold = colors[pei]
                        pnew = random.randint(0, NUM_COLORS - 1)
                        if pnew != pold:
                            # Update counts
                            color_counts[pold] -= 1
                            color_counts[pnew] += 1
                            colors[pei] = pnew
                            for ci in _EDGE_TO_CYCLES[pei]:
                                od = cycle_distinct[ci]
                                cycle_color_counts[ci][pold] -= 1
                                if cycle_color_counts[ci][pold] == 0:
                                    td = od - 1
                                else:
                                    td = od
                                onc = cycle_color_counts[ci][pnew]
                                cycle_color_counts[ci][pnew] += 1
                                if onc == 0:
                                    nd = td + 1
                                else:
                                    nd = td
                                cycle_distinct[ci] = nd
                            # Update violations
                            for ci in _EDGE_TO_CYCLES[pei]:
                                pass  # violations tracking is approximate during perturbation
                    # Recalculate violations
                    violations = sum(1 for d in cycle_distinct if d == 4)
                    num_distinct = sum(1 for c in color_counts if c > 0)
                    # Reset temperature
                    T = T_start * 0.5
        
    return best_colors
