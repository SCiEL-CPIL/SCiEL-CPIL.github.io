import time
import random

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

def find_rainbow_cycle(colors):
    """Find one rainbow 4-cycle. Returns None if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return cyc
    return None

def repair(colors):
    """
    Repair an invalid coloring by recoloring edges in rainbow cycles
    to match a neighbor's color. Returns the repaired valid coloring.
    """
    colors = colors[:]
    # Limit iterations to prevent infinite loops in pathological cases
    for _ in range(1000):
        cyc = find_rainbow_cycle(colors)
        if cyc is None:
            break
        # Pick the first edge in the cycle and change its color to the second edge's color
        # This ensures the cycle is no longer rainbow
        e0 = cyc[0]
        e1 = cyc[1]
        colors[e0] = colors[e1]
    return colors

def greedy_coloring(edge_order, rng):
    """
    Greedy: process edges in given order, assign each edge a color that
    keeps all 4-cycles valid. Try to maximize distinct colors by preferring
    unused colors when possible.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        # Try colors in an order that favors diversity
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        all_candidates = unused + used_list
        
        found = False
        for c in all_candidates:
            colors[e] = c
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                    continue
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            if valid:
                found = True
                break
        
        if not found:
            colors[e] = 0
        else:
            used_colors.add(colors[e])
    
    return colors

def local_search(colors, rng, deadline, tabu_list=None):
    """
    Local search with Tabu.
    Tries to increase the number of distinct colors.
    """
    if tabu_list is None:
        tabu_list = []
        
    colors = colors[:]
    best_count = count_colors(colors)
    current_count = best_count
    used_colors = set(colors)
    
    no_improve_steps = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        e = rng.randint(0, NUM_EDGES - 1)
        old = colors[e]
        
        # Generate candidates: new colors first, then random used colors
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        if unused:
            rng.shuffle(unused)
            candidates = unused[:10]
        else:
            candidates = [rng.randint(0, MAX_COLOR) for _ in range(10)]
            
        # Also try swapping with a random edge's color? No, just single edge change for now.
        
        found_improvement = False
        for new_c in candidates:
            if new_c == old:
                continue
            # Check if move is tabu
            if (e, old) in tabu_list:
                continue
                
            colors[e] = new_c
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            
            if valid:
                # Calculate new count
                old_count = current_count
                if new_c not in used_colors:
                    used_colors.add(new_c)
                    current_count += 1
                # Check if old color is still used
                # This is expensive, so we only do it if count might change
                # But we can just recalculate if we accept
                # For now, accept and update count properly
                # Actually, let's just accept and update
                if current_count > best_count:
                    best_count = current_count
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
                # Update tabu
                tabu_list.append((e, old))
                if len(tabu_list) > 20:
                    tabu_list.pop(0)
                
                found_improvement = True
                break
            else:
                colors[e] = old
                # Revert count change if we had tentatively changed it? 
                # We didn't change used_colors yet in the valid check path if invalid.
                # Wait, I added new_c to used_colors before checking validity? No.
                # In the code above, I added it inside the `if valid` block.
                # So if invalid, used_colors is unchanged.
        
        if not found_improvement:
             # Try a random valid move that doesn't increase count, to escape plateau?
             # For now, just break if no improvement
             pass

    return colors, best_count

def construct():
    start = time.time()
    deadline = start + 55
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # 1. Generate initial valid coloring using Greedy
    edge_indices = list(range(NUM_EDGES))
    orderings = [
        list(edge_indices),
        list(reversed(edge_indices)),
    ]
    for _ in range(5):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
        
    candidates = []
    for order in orderings:
        if time.time() >= deadline - 20:
            break
        c = greedy_coloring(order, rng)
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
            
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        best_count = candidates[0][0]
        best_colors = candidates[0][1][:]
    else:
        # Fallback
        best_colors = repair([rng.randint(0, MAX_COLOR) for _ in range(NUM_EDGES)])
        best_count = count_colors(best_colors)

    # 2. Iterated Local Search
    current_colors = best_colors[:]
    current_count = best_count
    
    for iteration in range(50):
        if time.time() >= deadline - 5:
            break
        
        # Perturb
        perturbed = current_colors[:]
        k = rng.randint(3, 10)
        for _ in range(k):
            e = rng.randint(0, NUM_EDGES - 1)
            perturbed[e] = rng.randint(0, MAX_COLOR)
            
        # Repair
        repaired = repair(perturbed)
        
        # Local Search on repaired
        ls_colors, ls_count = local_search(repaired, rng, deadline - 5)
        
        if ls_count > best_count:
            best_count = ls_count
            best_colors = ls_colors[:]
            current_colors = ls_colors[:]
        
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
        
    return best_colors
