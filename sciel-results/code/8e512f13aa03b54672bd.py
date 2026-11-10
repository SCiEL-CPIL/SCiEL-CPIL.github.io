"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_edge_list():
    """List of edges as (i,j) pairs in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a list of 4 edge indices.
    Return list of cycles (each is a tuple of 4 edge indices).
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

def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct colors."""
    violations = 0
    for cycle in _CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def construct():
    # Start with random high-diversity assignment
    random.seed(42)
    colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
    
    best_colors = colors[:]
    best_score = _count_violations(colors)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 30000
    
    current_violations = _count_violations(colors)
    current_distinct = len(set(colors))
    
    for iteration in range(max_iter):
        if T < T_min:
            break
        
        # Pick a random edge to recolor
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Try a new color - prefer colors already in use or nearby
        if random.random() < 0.7:
            new_color = random.choice(colors)
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        colors[ei] = new_color
        
        # Compute delta violations
        # Only cycles involving edge ei can change status
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            cycle = _CYCLES[ci]
            old_colors = [colors[e] if e != ei else old_color for e in cycle]
            new_colors = [colors[e] for e in cycle]
            old_bad = len(set(old_colors)) == 4
            new_bad = len(set(new_colors)) == 4
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        new_violations = current_violations + delta_violations
        new_distinct = len(set(colors))
        
        # Score: violations (lower better), then -distinct (higher distinct better)
        # Accept if violations decrease, or with probability based on temperature
        if new_violations < current_violations:
            accept = True
        elif new_violations == current_violations:
            # Prefer more distinct colors
            if new_distinct > current_distinct:
                accept = True
            else:
                # Accept with some probability to escape local minima
                accept = random.random() < 0.5
        else:
            # Violations increased - accept with probability exp(-delta/T)
            delta = new_violations - current_violations
            accept = random.random() < math.exp(-delta / T)
        
        if accept:
            current_violations = new_violations
            current_distinct = new_distinct
            if new_violations < best_score:
                best_score = new_violations
                best_colors = colors[:]
        else:
            colors[ei] = old_color
        
        T *= cooling
    
    # If we have violations, try to fix them with a more targeted approach
    if best_score > 0:
        # Try to fix violations by recoloring edges in violating cycles
        current_violations = best_score
        for _ in range(10000):
            # Find a violating cycle
            violating_cycle = None
            for cycle in _CYCLES:
                c0, c1, c2, c3 = best_colors[cycle[0]], best_colors[cycle[1]], best_colors[cycle[2]], best_colors[cycle[3]]
                if len({c0, c1, c2, c3}) == 4:
                    violating_cycle = cycle
                    break
            if violating_cycle is None:
                break
            
            # Try to recolor one of the edges in this cycle to match another
            for ei in violating_cycle:
                old_color = best_colors[ei]
                # Try to set it to match one of the other edges in the cycle
                for other_ei in violating_cycle:
                    if other_ei == ei:
                        continue
                    new_color = best_colors[other_ei]
                    best_colors[ei] = new_color
                    new_violations = _count_violations(best_colors)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        break
                    else:
                        best_colors[ei] = old_color
            
            if current_violations == 0:
                break
    
    return best_colors
