import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _is_rainbow(cyc, colors):
    """Check if the cycle has 4 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # State
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Helper to check if changing edge e to new_color creates any violation
    # Returns True if the move is valid (no new rainbow cycles)
    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
            
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            # Get current colors
            c0, c1, c2, c3 = colors_arr[cyc[0]], colors_arr[cyc[1]], colors_arr[cyc[2]], colors_arr[cyc[3]]
            
            # Check if current cycle is a rainbow
            # If it is already a rainbow, the state is invalid, but we assume we start valid.
            # We only care if the change *creates* a new rainbow.
            
            # Determine new colors
            nc0, nc1, nc2, nc3 = c0, c1, c2, c3
            if cyc[0] == e:
                nc0 = new_color
            elif cyc[1] == e:
                nc1 = new_color
            elif cyc[2] == e:
                nc2 = new_color
            elif cyc[3] == e:
                nc3 = new_color
            
            # If the cycle was NOT a rainbow before, check if it becomes one
            # If it WAS a rainbow before, we are already in an invalid state (shouldn't happen in our SA)
            # But to be safe, we check if the NEW state is a rainbow.
            # Actually, we only accept moves that keep the state valid.
            # So if the current state is valid, all cycles are non-rainbow.
            # We check if the new state makes this cycle a rainbow.
            
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        
        return True

    # Phase 1: Greedy Diversity
    # Start with all 0s. Try to add colors 1..119.
    # For each new color, find an edge to assign it.
    # We prefer edges that are "central" or have low impact? 
    # Actually, any edge that allows the new color without creating a rainbow.
    
    for new_c in range(1, MAX_COLOR + 1):
        if time.time() > deadline:
            break
            
        # Try edges in random order
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        assigned = False
        for e in edge_order:
            if _is_move_valid(e, new_c, colors):
                colors[e] = new_c
                assigned = True
                break
        
        if not assigned:
            continue

    # Phase 2: Simulated Annealing
    # Objective: Maximize distinct colors.
    # Constraint: No rainbow cycles.
    
    T = 1.0
    cooling = 0.99995
    min_T = 0.01
    
    current_distinct = len(set(colors))
    best_distinct = current_distinct
    best_colors = colors.copy()
    
    # Color counts for quick distinct update
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
        
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        
        # Pick a new color
        # Bias towards unused colors to increase diversity
        if np.random.random() < 0.3:
            unused = [c for c in range(MAX_COLOR + 1) if color_counts[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = np.random.randint(0, MAX_COLOR + 1)
        else:
            new_color = np.random.randint(0, MAX_COLOR + 1)
            
        if new_color == old_color:
            continue
            
        # Check validity
        if not _is_move_valid(e_idx, new_color, colors):
            continue
            
        # Calculate delta in distinct colors
        old_distinct = current_distinct
        # Apply move temporarily to calculate new distinct? 
        # Or just update counts.
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        new_distinct = old_distinct
        if color_counts[old_color] == 0:
            new_distinct -= 1
        if color_counts[new_color] == 1:
            new_distinct += 1
            
        # Acceptance criterion
        delta = new_distinct - old_distinct
        if delta >= 0 or np.random.random() < np.exp(delta / T):
            # Accept
            colors[e_idx] = new_color
            current_distinct = new_distinct
            
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = colors.copy()
        else:
            # Revert
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
            
        T *= cooling
        
    # Final check
    # Verify best_colors is valid
    valid = True
    for cyc in CYCLES:
        if _is_rainbow(cyc, best_colors):
            valid = False
            break
            
    if not valid:
        # Fallback to all zeros (always valid)
        return [0] * NUM_EDGES
        
    return best_colors.tolist()
