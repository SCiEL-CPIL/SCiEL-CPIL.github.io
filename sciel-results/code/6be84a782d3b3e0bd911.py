import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def greedy_coloring(edge_order, rng):
    """
    Greedy: process edges in given order, assign each edge a color that
    keeps all 4-cycles valid. Try to maximize distinct colors by preferring
    unused colors when possible.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        # Find all 4-cycles that contain edge e and are now "complete"
        candidate_colors = []
        
        # Try colors in an order that favors diversity
        # First try unused colors (to increase distinct count)
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        # Then try used colors
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        all_candidates = unused + used_list
        
        found = False
        for c in all_candidates:
            colors[e] = c
            # Check all cycles containing e
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                # Only check if all 4 edges in the cycle are assigned
                if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                    continue
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            if valid:
                found = True
                break
        
        if not found:
            # Shouldn't happen, but fallback to 0
            colors[e] = 0
        else:
            used_colors.add(colors[e])
    
    return colors

def simulated_annealing(initial_colors, time_limit, rng):
    """
    Simulated Annealing to maximize distinct colors while maintaining validity.
    """
    colors = initial_colors[:]
    used_colors = set(colors)
    num_distinct = len(used_colors)
    
    # Calculate initial violations
    num_violations = 0
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            num_violations += 1
            
    # Cost function: We want to minimize violations and maximize distinct colors.
    # Let's define cost = 1000 * num_violations - num_distinct
    # If num_violations > 0, cost is high.
    # If num_violations == 0, cost = -num_distinct. Lower is better.
    
    current_cost = 1000 * num_violations - num_distinct
    
    best_colors = colors[:]
    best_cost = current_cost
    best_valid_count = num_distinct if num_violations == 0 else -1
    
    start_time = time.time()
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() < time_limit:
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Pick a new color
        # Bias towards unused colors to try to increase diversity
        if rng.random() < 0.5:
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, MAX_COLOR)
        else:
            new_color = rng.randint(0, MAX_COLOR)
            
        if new_color == old_color:
            continue
            
        # Calculate delta cost
        # We need to check all cycles containing edge e
        delta_violations = 0
        for cyc in _CYCLES_BY_EDGE[e]:
            was_valid = is_valid_cycle(colors, cyc)
            # Temporarily change color
            colors[e] = new_color
            is_valid_now = is_valid_cycle(colors, cyc)
            # Restore color for next cycle check? No, we can just track state.
            # Actually, is_valid_cycle reads from `colors`.
            # So we must be careful.
            
            if was_valid and not is_valid_now:
                delta_violations += 1
            elif not was_valid and is_valid_now:
                delta_violations -= 1
            
            # Restore old color to check next cycle?
            # No, the cycle check depends on all 4 edges.
            # If we change e, we affect all cycles containing e.
            # The other edges in the cycle are unchanged.
            # So we can just check with new_color in place.
            # But we need to restore it if we reject?
            # Yes.
            
            # Wait, I modified `colors[e]` in place.
            # I should restore it after checking all cycles for this move.
        
        # Restore old color for now, we'll decide later
        colors[e] = old_color
        
        # Calculate delta distinct colors
        # If new_color is not in used_colors, distinct +1
        # If old_color was the only one of its kind, distinct -1
        # We need to count occurrences.
        # This is expensive to do every step.
        # Let's maintain a count array.
        
        # Optimization: Maintain color_counts
        # Let's refactor to use a helper class or just local variables in the loop.
        # For simplicity and speed, let's just recalculate distinct count if the move is accepted?
        # No, that's O(N).
        # Let's maintain `color_counts` array.
        
        # Re-implementation with color_counts
        pass
        
    return best_colors, best_valid_count

def local_improve_aggressive_v2(colors, rng, deadline):
    """
    Improved local search using Simulated Annealing principles.
    """
    colors = colors[:]
    used_colors = set(colors)
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
        
    num_violations = 0
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            num_violations += 1
            
    num_distinct = len(used_colors)
    current_cost = 1000 * num_violations - num_distinct
    
    best_colors = colors[:]
    best_valid_count = num_distinct if num_violations == 0 else -1
    
    T = 5.0
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Adaptive cooling
        if T > 0.1:
            T *= cooling_rate
        else:
            # Restart temperature if stuck? Or just keep low
            T = 0.1
            
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose new color
        # 50% chance to pick an unused color if available
        if rng.random() < 0.5 and len(used_colors) < MAX_COLOR + 1:
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, MAX_COLOR)
        else:
            new_color = rng.randint(0, MAX_COLOR)
            
        if new_color == old_color:
            continue
            
        # Evaluate move
        delta_violations = 0
        for cyc in _CYCLES_BY_EDGE[e]:
            was_valid = is_valid_cycle(colors, cyc)
            colors[e] = new_color
            is_valid_now = is_valid_cycle(colors, cyc)
            colors[e] = old_color # Restore immediately for next check? 
            # Actually, if we restore immediately, the next cycle check will see old_color.
            # But the next cycle also contains e. So it should see new_color.
            # So we must keep new_color while checking all cycles containing e.
            
        # Corrected Logic:
        colors[e] = old_color
        delta_violations = 0
        for cyc in _CYCLES_BY_EDGE[e]:
            was_valid = is_valid_cycle(colors, cyc)
            colors[e] = new_color
            is_valid_now = is_valid_cycle(colors, cyc)
            if was_valid and not is_valid_now:
                delta_violations += 1
            elif not was_valid and is_valid_now:
                delta_violations -= 1
        colors[e] = old_color # Restore
        
        # Calculate delta distinct
        delta_distinct = 0
        if new_color not in used_colors:
            delta_distinct += 1
        if old_color in used_colors:
            if color_counts[old_color] == 1:
                delta_distinct -= 1
        
        delta_cost = 1000 * delta_violations - delta_distinct
        
        # Acceptance
        if delta_cost <= 0 or (T > 0 and rng.random() < math.exp(-delta_cost / T)):
            # Accept move
            colors[e] = new_color
            
            # Update state
            if new_color not in used_colors:
                used_colors.add(new_color)
                color_counts[new_color] = 1
            else:
                color_counts[new_color] += 1
                
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                del color_counts[old_color]
                used_colors.discard(old_color)
                
            num_violations += delta_violations
            num_distinct += delta_distinct
            current_cost += delta_cost
            
            # Update best
            if num_violations == 0 and num_distinct > best_valid_count:
                best_valid_count = num_distinct
                best_colors = colors[:]
                
        # Occasional restart if stuck
        if T < 0.5 and rng.random() < 0.01:
             # Perturb
             for _ in range(5):
                 pe = rng.randint(0, NUM_EDGES - 1)
                 pc = rng.randint(0, MAX_COLOR)
                 # Just apply blindly? No, check validity?
                 # Perturbation should be random.
                 # Let's just reset T
             T = 5.0

    return best_colors, best_valid_count

def construct():
    start = time.time()
    deadline = start + 55
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Generate initial candidates
    candidates = []
    
    # 1. Greedy with different orderings
    edge_indices = list(range(NUM_EDGES))
    
    orderings = [
        list(edge_indices),
        list(reversed(edge_indices)),
        sorted(edge_indices, key=lambda e: len(_CYCLES_BY_EDGE[e])),
        sorted(edge_indices, key=lambda e: -len(_CYCLES_BY_EDGE[e]))
    ]
    
    for _ in range(5):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
        
    for order in orderings:
        if time.time() > deadline - 10:
            break
        c = greedy_coloring(order, rng)
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
            
    if not candidates:
        # Fallback
        candidates.append((1, [0] * NUM_EDGES))
        
    # Sort by count
    candidates.sort(key=lambda x: -x[0])
    
    # Run SA on top candidates
    for cnt, c in candidates[:5]:
        if time.time() > deadline - 5:
            break
        improved_c, improved_cnt = local_improve_aggressive_v2(c, rng, deadline - 2)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c[:]
            
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
        
    return best_colors
