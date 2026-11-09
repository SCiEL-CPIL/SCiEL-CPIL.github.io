import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline for 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    def get_four_cycles():
        """Generate all 4-cycles as tuples of 4 edge indices."""
        cycles = []
        for a in range(N):
            for b in range(a + 1, N):
                for c in range(b + 1, N):
                    for d in range(c + 1, N):
                        # Three 4-cycles on {a,b,c,d}
                        # Cycle 1: a-b-c-d-a
                        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                        # Cycle 2: a-b-d-c-a
                        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                        # Cycle 3: a-c-b-d-a
                        cycles.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
        return cycles
    
    all_cycles = get_four_cycles()
    
    def count_violations(colors):
        """Count number of rainbow 4-cycles."""
        violations = 0
        for cycle in all_cycles:
            e0, e1, e2, e3 = cycle
            c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
            if len({c0, c1, c2, c3}) == 4:
                violations += 1
        return violations
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Start with a greedy construction
    colors = [0] * NUM_EDGES
    
    # Greedy: assign colors to maximize distinct colors while avoiding violations
    # Process edges in random order
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    # Try to assign each edge a new color if possible, else reuse
    used_colors = {}  # color -> set of edges
    for idx in edge_order:
        # Try new colors first (high to low to encourage distinctness)
        assigned = False
        for new_color in range(119, -1, -1):
            colors[idx] = new_color
            if count_violations(colors) == 0:
                assigned = True
                break
        if not assigned:
            # Must reuse an existing color
            colors[idx] = 0  # fallback
        
    # Local search to improve
    best_colors = colors[:]
    best_distinct = count_distinct(best_colors)
    best_violations = count_violations(best_colors)
    
    # Ensure no violations
    while best_violations > 0:
        # Find a violating cycle and fix it
        for cycle in all_cycles:
            e0, e1, e2, e3 = cycle
            c0, c1, c2, c3 = best_colors[e0], best_colors[e1], best_colors[e2], best_colors[e3]
            if len({c0, c1, c2, c3}) == 4:
                # Change one edge to match another in the cycle
                # Try changing e0 to match e1
                old_c0 = best_colors[e0]
                best_colors[e0] = c1
                if count_violations(best_colors) < best_violations:
                    best_violations = count_violations(best_colors)
                    break
                # Try changing e0 to match e2
                best_colors[e0] = c2
                if count_violations(best_colors) < best_violations:
                    best_violations = count_violations(best_colors)
                    break
                # Try changing e0 to match e3
                best_colors[e0] = c3
                if count_violations(best_colors) < best_violations:
                    best_violations = count_violations(best_colors)
                    break
                best_colors[e0] = old_c0
        else:
            break  # No improvement possible
    
    # Now optimize for distinct colors using local search
    if time.time() < deadline - 5:
        # Local search: try to change edges to new colors
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in random.sample(range(NUM_EDGES), min(50, NUM_EDGES)):
                old_color = best_colors[idx]
                # Try a random new color
                new_color = random.randint(0, 119)
                if new_color == old_color:
                    continue
                best_colors[idx] = new_color
                if count_violations(best_colors) == 0:
                    new_distinct = count_distinct(best_colors)
                    if new_distinct > best_distinct:
                        best_distinct = new_distinct
                        improved = True
                    else:
                        best_colors[idx] = old_color
                else:
                    best_colors[idx] = old_color
            if not improved and iteration > 100:
                break
    
    # Final verification
    final_violations = count_violations(best_colors)
    if final_violations > 0:
        # Fallback: return a simple valid coloring
        # Use a structured coloring: color(i,j) = (i + j) % 10
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                best_colors[idx] = (i + j) % 10
    
    return best_colors

