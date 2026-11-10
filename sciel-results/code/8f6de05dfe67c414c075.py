import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles on a,b,c,d
                    # a-b-c-d-a
                    cycles.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                    # a-b-d-c-a
                    cycles.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                    # a-c-b-d-a
                    cycles.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))
    return cycles

def _count_colors(colors):
    return len(set(colors))

def _count_violations(colors, cycles):
    """Count how many 4-cycles have 4 distinct colors (violations)."""
    count = 0
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            count += 1
    return count

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    cycles = _build_cycles()
    num_cycles = len(cycles)
    
    # Precompute: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        edge_to_cycles[e1].append(ci)
        edge_to_cycles[e2].append(ci)
        edge_to_cycles[e3].append(ci)
        edge_to_cycles[e4].append(ci)
    
    best_colors = None
    best_score = 0
    best_violations = float('inf')
    
    random.seed(42)
    
    # Multiple random restarts with local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize: random coloring with limited palette
        # Start with a small number of colors to ensure validity, then expand
        init_palette_size = random.choice([2, 3, 4, 5])
        colors = [random.randint(0, init_palette_size - 1) for _ in range(NUM_EDGES)]
        
        violations = _count_violations(colors, cycles)
        score = _count_colors(colors)
        
        # Local search
        stuck_iterations = 0
        max_stuck = 5000
        
        while time.time() < deadline - 3.0 and stuck_iterations < max_stuck:
            # Choose move type based on state
            if violations == 0:
                # Valid state: try to increase diversity
                # Pick an edge and change to a new color (not currently used)
                used_colors = set(colors)
                new_colors = [c for c in range(120) if c not in used_colors]
                if not new_colors:
                    break
                
                # Targeted: pick edge that appears in fewest cycles (less impact)
                # Or pick random edge
                if random.random() < 0.7:
                    # Pick edge with minimum cycle count among those that can take new color
                    candidates = list(range(NUM_EDGES))
                    random.shuffle(candidates)
                    # Try a few candidates
                    improved = False
                    for edge_idx in candidates[:20]:
                        old_color = colors[edge_idx]
                        new_color = random.choice(new_colors)
                        colors[edge_idx] = new_color
                        # Check if still valid (only need to check cycles containing this edge)
                        still_valid = True
                        for ci in edge_to_cycles[edge_idx]:
                            e1, e2, e3, e4 = cycles[ci]
                            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                            if len({c1, c2, c3, c4}) == 4:
                                still_valid = False
                                break
                        if still_valid:
                            new_score = _count_colors(colors)
                            if new_score > score:
                                score = new_score
                                stuck_iterations = 0
                                improved = True
                                break
                            else:
                                colors[edge_idx] = old_color
                        else:
                            colors[edge_idx] = old_color
                    
                    if not improved:
                        stuck_iterations += 1
                else:
                    # Random perturbation
                    edge_idx = random.randint(0, NUM_EDGES - 1)
                    old_color = colors[edge_idx]
                    new_color = random.choice(new_colors)
                    colors[edge_idx] = new_color
                    still_valid = True
                    for ci in edge_to_cycles[edge_idx]:
                        e1, e2, e3, e4 = cycles[ci]
                        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                        if len({c1, c2, c3, c4}) == 4:
                            still_valid = False
                            break
                    if still_valid:
                        new_score = _count_colors(colors)
                        if new_score > score:
                            score = new_score
                            stuck_iterations = 0
                        else:
                            colors[edge_idx] = old_color
                            stuck_iterations += 1
                    else:
                        colors[edge_idx] = old_color
                        stuck_iterations += 1
            else:
                # Invalid state: try to repair
                # Pick a violated cycle and change one of its edges to match another
                # Find a violated cycle
                violated_cycle_idx = -1
                for ci in range(num_cycles):
                    e1, e2, e3, e4 = cycles[ci]
                    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                    if len({c1, c2, c3, c4}) == 4:
                        violated_cycle_idx = ci
                        break
                
                if violated_cycle_idx == -1:
                    # Shouldn't happen, but just in case
                    violations = 0
                    continue
                
                e1, e2, e3, e4 = cycles[violated_cycle_idx]
                c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                
                # Try changing one edge to match another in the cycle
                # Pick which edge to change (random)
                edge_positions = [e1, e2, e3, e4]
                edge_colors = [c1, c2, c3, c4]
                change_pos = random.randint(0, 3)
                change_edge = edge_positions[change_pos]
                
                # Try matching to one of the other edges
                for match_pos in [p for p in range(4) if p != change_pos]:
                    match_edge = edge_positions[match_pos]
                    new_color = colors[match_edge]
                    
                    old_color = colors[change_edge]
                    colors[change_edge] = new_color
                    
                    # Check if this reduces violations (check cycles containing change_edge)
                    old_violations = violations
                    new_violations = old_violations
                    # Recalculate violations for affected cycles
                    for ci in edge_to_cycles[change_edge]:
                        ee1, ee2, ee3, ee4 = cycles[ci]
                        cc1, cc2, cc3, cc4 = colors[ee1], colors[ee2], colors[ee3], colors[ee4]
                        is_violated = len({cc1, cc2, cc3, cc4}) == 4
                        if is_violated:
                            new_violations += 1
                    
                    if new_violations < old_violations:
                        violations = new_violations
                        stuck_iterations = 0
                        break
                    elif new_violations == old_violations and random.random() < 0.3:
                        # Accept with some probability to escape local minima
                        violations = new_violations
                        stuck_iterations = 0
                        break
                    else:
                        colors[change_edge] = old_color
                
                else:
                    stuck_iterations += 1
        
        # End of local search for this restart
        current_score = _count_colors(colors)
        current_violations = _count_violations(colors, cycles)
        
        if current_violations == 0 and current_score > best_score:
            best_score = current_score
            best_colors = colors[:]
            best_violations = 0
        
        # Track best even if not fully valid (for debugging, but we only return valid)
        if current_violations < best_violations:
            best_violations = current_violations
    
    if best_colors is None:
        # Fallback: all zeros
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    return best_colors

# EVOLVE-BLOCK-START
def construct():
    # Use the helper defined above
    return _construct_impl()
# EVOLVE-BLOCK-END
