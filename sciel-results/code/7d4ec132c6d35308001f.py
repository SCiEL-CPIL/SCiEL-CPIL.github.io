"""Program for the edge-coloring task: maximize distinct colors in K16 with no rainbow 4-cycle."""
import time
import random
import itertools
import numpy as np

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + j - i - 1

def _precompute_cycles():
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        cycles.append([_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)])
        cycles.append([_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(c,a)])
        cycles.append([_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(d,a)])
    return cycles

def _precompute_edge_cycles():
    all_cycles = _precompute_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return all_cycles, edge_to_cycles

def _check_cycle(cyc, colors, changed_edge, new_color):
    """Check if a single cycle is valid after changing changed_edge to new_color."""
    c0 = new_color if cyc[0] == changed_edge else colors[cyc[0]]
    c1 = new_color if cyc[1] == changed_edge else colors[cyc[1]]
    c2 = new_color if cyc[2] == changed_edge else colors[cyc[2]]
    c3 = new_color if cyc[3] == changed_edge else colors[cyc[3]]
    # Check if all 4 are distinct
    if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
        return True
    return False

def _is_valid_change(colors, edge_idx, new_color, all_cycles, edge_to_cycles):
    for ci in edge_to_cycles[edge_idx]:
        if not _check_cycle(all_cycles[ci], colors, edge_idx, new_color):
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_coloring(all_cycles, edge_to_cycles, rng, deadline):
    """Greedy coloring with random edge order."""
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    edge_order = list(range(NUM_EDGES))
    rng.shuffle(edge_order)
    
    current_count = 1
    for e in edge_order:
        if time.time() > deadline:
            break
        # Try to assign a new color
        new_color = current_count
        if new_color >= NUM_COLORS:
            break
        if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
            colors[e] = new_color
            current_count += 1
    
    return colors, current_count

def _local_search(colors, all_cycles, edge_to_cycles, rng, deadline, max_iter=5000):
    """Local search to improve color count."""
    colors = colors[:]
    current_count = _count_colors(colors)
    used = set(colors)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to change to a new unused color
        if len(used) < NUM_COLORS:
            new_color = rng.randint(0, NUM_COLORS - 1)
            if new_color not in used:
                if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
                    colors[e] = new_color
                    used.add(new_color)
                    current_count += 1
                    continue
        
        # Try to change to any color (might allow further changes later)
        new_color = rng.randint(0, NUM_COLORS - 1)
        if new_color != old_color:
            if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
                colors[e] = new_color
                # Update used set
                # Check if old color is still used
                still_used = False
                for i in range(NUM_EDGES):
                    if i != e and colors[i] == old_color:
                        still_used = True
                        break
                if not still_used:
                    used.discard(old_color)
                if new_color not in used:
                    used.add(new_color)
    
    return colors, current_count

def _hill_climbing_with_swaps(colors, all_cycles, edge_to_cycles, rng, deadline, max_iter=10000):
    """Hill climbing with swap moves to escape local optima."""
    colors = colors[:]
    current_count = _count_colors(colors)
    used = set(colors)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick two random edges
        e1 = rng.randint(0, NUM_EDGES - 1)
        e2 = rng.randint(0, NUM_EDGES - 1)
        if e1 == e2:
            continue
        
        c1 = colors[e1]
        c2 = colors[e2]
        if c1 == c2:
            continue
        
        # Swap colors
        colors[e1] = c2
        colors[e2] = c1
        
        # Check validity
        valid = True
        for ci in edge_to_cycles[e1]:
            if not _check_cycle(all_cycles[ci], colors, e1, c2):
                valid = False
                break
        if valid:
            for ci in edge_to_cycles[e2]:
                if not _check_cycle(all_cycles[ci], colors, e2, c1):
                    valid = False
                    break
        
        if valid:
            # Count new colors
            new_used = set(colors)
            new_count = len(new_used)
            if new_count >= current_count:
                current_count = new_count
                used = new_used
            else:
                # Revert
                colors[e1] = c1
                colors[e2] = c2
        else:
            # Revert
            colors[e1] = c1
            colors[e2] = c2
    
    return colors, current_count

def _simulated_annealing(colors, all_cycles, edge_to_cycles, rng, deadline, max_iter=20000):
    """Simulated annealing to escape local optima."""
    colors = colors[:]
    current_count = _count_colors(colors)
    used = set(colors)
    
    temperature = 10.0
    cooling_rate = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        new_color = rng.randint(0, NUM_COLORS - 1)
        
        if new_color == old_color:
            continue
        
        # Try the change
        colors[e] = new_color
        
        # Check validity
        valid = True
        for ci in edge_to_cycles[e]:
            if not _check_cycle(all_cycles[ci], colors, e, new_color):
                valid = False
                break
        
        if not valid:
            # Revert
            colors[e] = old_color
            continue
        
        # Evaluate
        new_used = set(colors)
        new_count = len(new_used)
        delta = new_count - current_count
        
        # Accept if better, or with probability based on temperature
        if delta >= 0:
            current_count = new_count
            used = new_used
        else:
            # Accept with probability exp(delta / temperature)
            if rng.random() < np.exp(delta / temperature):
                current_count = new_count
                used = new_used
            else:
                # Revert
                colors[e] = old_color
        
        temperature *= cooling_rate
    
    return colors, current_count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_cycles, edge_to_cycles = _precompute_edge_cycles()
    
    best_count = 0
    best_colors = [0] * NUM_EDGES
    
    # Phase 1: Multiple greedy restarts
    for restart in range(300):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(42 + restart)
        colors, count = _greedy_coloring(all_cycles, edge_to_cycles, rng, deadline - 10.0)
        
        if count > best_count:
            best_count = count
            best_colors = colors[:]
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 5.0:
        rng = random.Random(12345)
        colors, count = _local_search(best_colors, all_cycles, edge_to_cycles, rng, deadline - 5.0, max_iter=3000)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
    
    # Phase 3: Hill climbing with swaps
    if best_count > 0 and time.time() < deadline - 3.0:
        rng = random.Random(54321)
        colors, count = _hill_climbing_with_swaps(best_colors, all_cycles, edge_to_cycles, rng, deadline - 3.0, max_iter=8000)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
    
    # Phase 4: Simulated annealing
    if best_count > 0 and time.time() < deadline - 1.0:
        rng = random.Random(99999)
        colors, count = _simulated_annealing(best_colors, all_cycles, edge_to_cycles, rng, deadline - 1.0, max_iter=15000)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
    
    return best_colors
