import time
import random
import itertools
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    """Build all 4-cycles and edge-to-cycle mappings."""
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        cycles.append((get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)))
        # Cycle 2: a-b-d-c-a
        cycles.append((get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)))
        # Cycle 3: a-c-b-d-a
        cycles.append((get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)))
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(idx)
    
    return cycles, edge_to_cycles

def build_adjacency():
    """Build edge adjacency (edges sharing a vertex)."""
    edge_endpoints = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_endpoints.append((i, j))
    
    # Map vertex to edges incident to it
    vertex_to_edges = [[] for _ in range(N)]
    for idx, (u, v) in enumerate(edge_endpoints):
        vertex_to_edges[u].append(idx)
        vertex_to_edges[v].append(idx)
    
    # Build adjacency: two edges are adjacent if they share a vertex
    edge_adj = [set() for _ in range(NUM_EDGES)]
    for v in range(N):
        edges_at_v = vertex_to_edges[v]
        for i in range(len(edges_at_v)):
            for j in range(i + 1, len(edges_at_v)):
                e1, e2 = edges_at_v[i], edges_at_v[j]
                edge_adj[e1].add(e2)
                edge_adj[e2].add(e1)
    
    return edge_endpoints, edge_adj

def count_distinct(colors):
    return len(set(colors))

def is_valid_fast(colors, cycles):
    """Check validity using vectorized approach."""
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def count_invalid_cycles(colors, cycles):
    """Count how many cycles are invalid (have 4 distinct colors)."""
    invalid = 0
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            invalid += 1
    return invalid

def check_recolor_valid(colors, edge, new_color, cycles, edge_to_cycles):
    """Check if recoloring `edge` to `new_color` maintains validity.
    Only need to check cycles containing this edge."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    
    for cycle_idx in edge_to_cycles[edge]:
        e1, e2, e3, e4 = cycles[cycle_idx]
        # Get the four colors after recoloring
        c1 = new_color if e1 == edge else colors[e1]
        c2 = new_color if e2 == edge else colors[e2]
        c3 = new_color if e3 == edge else colors[e3]
        c4 = new_color if e4 == edge else colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def check_swap_valid(colors, edge1, edge2, cycles, edge_to_cycles):
    """Check if swapping colors of edge1 and edge2 maintains validity.
    Only need to check cycles containing either edge."""
    c1_old = colors[edge1]
    c2_old = colors[edge2]
    if c1_old == c2_old:
        return True
    
    # Get all cycles that contain either edge
    affected_cycles = set(edge_to_cycles[edge1]) | set(edge_to_cycles[edge2])
    
    for cycle_idx in affected_cycles:
        e1, e2, e3, e4 = cycles[cycle_idx]
        # Get the four colors after swap
        c1 = c2_old if e1 == edge1 else (c1_old if e1 == edge2 else colors[e1])
        c2 = c2_old if e2 == edge1 else (c1_old if e2 == edge2 else colors[e2])
        c3 = c2_old if e3 == edge1 else (c1_old if e3 == edge2 else colors[e3])
        c4 = c2_old if e4 == edge1 else (c1_old if e4 == edge2 else colors[e4])
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    cycles, edge_to_cycles = build_cycles()
    edge_endpoints, edge_adj = build_adjacency()
    
    # Precompute: for each edge, list of other edges in same cycle (for fast constraint checking)
    # This helps identify which edges can share colors to cover cycles
    
    best_distinct = 0
    best_colors = None
    
    def try_initialization(method, rng):
        """Try different initialization strategies."""
        if method == 0:
            # All same color
            return [0] * NUM_EDGES
        elif method == 1:
            # Random assignment from small palette
            palette_size = rng.randint(3, 8)
            return [rng.randint(0, palette_size - 1) for _ in range(NUM_EDGES)]
        elif method == 2:
            # Structured: color by distance in some ordering
            colors = [0] * NUM_EDGES
            for idx, (u, v) in enumerate(edge_endpoints):
                colors[idx] = (v - u) % 5
            return colors
        elif method == 3:
            # Random with bias toward few colors
            palette_size = rng.randint(2, 6)
            return [rng.choice(range(palette_size)) for _ in range(NUM_EDGES)]
        elif method == 4:
            # Greedy: try to assign new color, fall back to existing if invalid
            colors = [0] * NUM_EDGES
            edge_order = list(range(NUM_EDGES))
            rng.shuffle(edge_order)
            next_color = 1
            for e in edge_order:
                # Try new color
                colors[e] = next_color
                if check_recolor_valid(colors, e, next_color, cycles, edge_to_cycles):
                    next_color += 1
                else:
                    # Try existing colors
                    assigned = False
                    for c in range(next_color):
                        colors[e] = c
                        if check_recolor_valid(colors, e, c, cycles, edge_to_cycles):
                            assigned = True
                            break
                    if not assigned:
                        colors[e] = 0
            return colors
        else:
            # Random restart with medium palette
            palette_size = rng.randint(5, 15)
            return [rng.randint(0, palette_size - 1) for _ in range(NUM_EDGES)]
    
    def simulated_annealing(colors, max_iters, rng, T_start=1.0, T_end=0.01):
        """Run simulated annealing to maximize distinct colors."""
        current_colors = colors[:]
        current_distinct = count_distinct(current_colors)
        current_invalid = count_invalid_cycles(current_colors, cycles)
        
        best_local_distinct = current_distinct
        best_local_colors = current_colors[:]
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
        
        # Track used colors for efficient new color assignment
        color_counts = {}
        for c in current_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        max_color_used = max(current_colors) if current_colors else 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.6:
                # Single edge recolor
                e = rng.randint(0, NUM_EDGES - 1)
                old_color = current_colors[e]
                
                # Decide new color
                r = rng.random()
                if r < 0.5:
                    # Try a new color
                    new_color = max_color_used + 1
                elif r < 0.8:
                    # Try an existing color
                    existing = list(color_counts.keys())
                    if not existing:
                        new_color = 0
                    else:
                        new_color = rng.choice(existing)
                else:
                    # Random color
                    new_color = rng.randint(0, max_color_used + 5)
                
                if new_color == old_color:
                    continue
                
                # Check validity
                if not check_recolor_valid(current_colors, e, new_color, cycles, edge_to_cycles):
                    continue
                
                # Calculate delta in objective
                # Objective: maximize distinct, minimize invalid
                old_invalid = current_invalid
                old_distinct = current_distinct
                
                # Apply
                current_colors[e] = new_color
                
                # Update color counts
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
                if new_color > max_color_used:
                    max_color_used = new_color
                
                new_distinct = len(color_counts)
                
                # Recount invalid (expensive, but we can approximate)
                # For speed, only recount if distinct changed or we're improving
                if new_distinct != old_distinct:
                    new_invalid = count_invalid_cycles(current_colors, cycles)
                else:
                    new_invalid = old_invalid  # approximation
                
                # Acceptance criterion
                delta_distinct = new_distinct - old_distinct
                delta_invalid = new_invalid - old_invalid
                delta_obj = delta_distinct - 0.1 * delta_invalid
                
                if delta_obj >= 0 or rng.random() < pow(2.718, delta_obj / max(T, 1e-6)):
                    current_distinct = new_distinct
                    current_invalid = new_invalid
                    if current_distinct > best_local_distinct:
                        best_local_distinct = current_distinct
                        best_local_colors = current_colors[:]
                else:
                    # Revert
                    current_colors[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
                    if color_counts[new_color] == 0:
                        del color_counts[new_color]
                    if new_color == max_color_used:
                        # Find new max
                        if color_counts:
                            max_color_used = max(color_counts.keys())
            
            elif move_type < 0.9:
                # Swap colors between two non-adjacent edges
                e1 = rng.randint(0, NUM_EDGES - 1)
                e2 = rng.randint(0, NUM_EDGES - 1)
                if e1 == e2 or e2 in edge_adj[e1]:
                    continue
                
                c1 = current_colors[e1]
                c2 = current_colors[e2]
                if c1 == c2:
                    continue
                
                # Check validity of swap
                if not check_swap_valid(current_colors, e1, e2, cycles, edge_to_cycles):
                    continue
                
                # Apply swap
                current_colors[e1], current_colors[e2] = c2, c1
                
                # Distinct count doesn't change with a swap
                # But invalid count might
                new_invalid = count_invalid_cycles(current_colors, cycles)
                old_invalid = current_invalid
                
                delta_invalid = new_invalid - old_invalid
                delta_obj = -0.1 * delta_invalid
                
                if delta_obj >= 0 or rng.random() < pow(2.718, delta_obj / max(T, 1e-6)):
                    current_invalid = new_invalid
                else:
                    # Revert
                    current_colors[e1], current_colors[e2] = c1, c2
            
            else:
                # Recolor a random edge to a completely new color (aggressive exploration)
                e = rng.randint(0, NUM_EDGES - 1)
                old_color = current_colors[e]
                new_color = max_color_used + 1
                
                if not check_recolor_valid(current_colors, e, new_color, cycles, edge_to_cycles):
                    continue
                
                current_colors[e] = new_color
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = 1
                max_color_used = new_color
                
                new_distinct = len(color_counts)
                new_invalid = count_invalid_cycles(current_colors, cycles)
                
                delta_distinct = new_distinct - current_distinct
                delta_invalid = new_invalid - current_invalid
                delta_obj = delta_distinct - 0.1 * delta_invalid
                
                if delta_obj >= 0 or rng.random() < pow(2.718, delta_obj / max(T, 1e-6)):
                    current_distinct = new_distinct
                    current_invalid = new_invalid
                    if current_distinct > best_local_distinct:
                        best_local_distinct = current_distinct
                        best_local_colors = current_colors[:]
                else:
                    # Revert
                    current_colors[e] = old_color
                    color_counts[old_color] += 1
                    del color_counts[new_color]
                    if new_color == max_color_used:
                        if color_counts:
                            max_color_used = max(color_counts.keys())
            
            T *= cooling
        
        return best_local_colors, best_local_distinct
    
    # Multi-restart simulated annealing
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts * 7919 + 13)
        
        method = num_restarts % 5
        colors = try_initialization(method, rng)
        
        # Initial repair: fix any invalid cycles
        invalid = count_invalid_cycles(colors, cycles)
        repair_iters = 0
        while invalid > 0 and repair_iters < 1000 and time.time() < deadline:
            repair_iters += 1
            # Find an invalid cycle and fix it
            fixed = False
            for e1, e2, e3, e4 in cycles:
                c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                    # This cycle is invalid. Pick an edge and match its color to another in the cycle.
                    # Try to match e1 to e2, e3, or e4
                    for target_e in [e2, e3, e4]:
                        target_c = colors[target_e]
                        if check_recolor_valid(colors, e1, target_c, cycles, edge_to_cycles):
                            colors[e1] = target_c
                            fixed = True
                            break
                    if fixed:
                        break
            if not fixed:
                # Fallback: set all four to same color
                colors[e1] = colors[e2] = colors[e3] = colors[e4] = 0
            invalid = count_invalid_cycles(colors, cycles)
        
        # Run simulated annealing
        max_iters = 20000
        sa_colors, sa_distinct = simulated_annealing(colors, max_iters, rng)
        
        if sa_distinct > best_distinct:
            best_distinct = sa_distinct
            best_colors = sa_colors[:]
        
        # If we've found a good solution, do more refinement
        if best_distinct >= 15:
            # Try more iterations on the best solution
            if best_colors is not None:
                rng2 = random.Random(num_restarts * 104729 + 7)
                refined, refined_distinct = simulated_annealing(best_colors, 30000, rng2, T_start=0.5, T_end=0.005)
                if refined_distinct > best_distinct:
                    best_distinct = refined_distinct
                    best_colors = refined[:]
        
        if time.time() > deadline - 5:
            break
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Distinct colors: {len(set(result))}")
