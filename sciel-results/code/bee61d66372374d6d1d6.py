"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations
import time

N = 16
E = N * (N - 1) // 2  # 120 edges

# Precompute edge index mapping
_edge_idx = {}
def _build_edge_idx():
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            _edge_idx[(i, j)] = idx
            idx += 1
_build_edge_idx()

def _get_edge(i, j):
    if i > j:
        i, j = j, i
    return _edge_idx[(i, j)]

# Precompute all 4-cycles: for each set of 4 vertices, there are 3 possible 4-cycles
# A 4-cycle on vertices {a,b,c,d} can be:
#   (a,b,c,d) - edges: ab, bc, cd, da
#   (a,b,d,c) - edges: ab, bd, dc, ca
#   (a,c,b,d) - edges: ac, cb, bd, da
# Each 4-cycle is defined by 4 edges. We need to check that no 4-cycle has 4 distinct colors.

# Precompute all 4-cycles as tuples of 4 edge indices
def _precompute_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = (_get_edge(a,b), _get_edge(b,c), _get_edge(c,d), _get_edge(d,a))
        cycles.append(e1)
        # Cycle 2: a-b-d-c-a
        e2 = (_get_edge(a,b), _get_edge(b,d), _get_edge(d,c), _get_edge(c,a))
        cycles.append(e2)
        # Cycle 3: a-c-b-d-a
        e3 = (_get_edge(a,c), _get_edge(c,b), _get_edge(b,d), _get_edge(d,a))
        cycles.append(e3)
    return cycles

_CYCLES = _precompute_cycles()
_NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles it belongs to
_EDGE_TO_CYCLES = [[] for _ in range(E)]
for ci, cyc in enumerate(_CYCLES):
    for e in cyc:
        _EDGE_TO_CYCLES[e].append(ci)

def _count_colors(colors):
    """Count number of distinct colors used."""
    return len(set(colors))

def _count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors (violations)."""
    violations = 0
    for cyc in _CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def _evaluate(colors):
    """Return (violations, num_colors). Lower violations is better, higher colors is better."""
    violations = _count_violations(colors)
    if violations > 0:
        return violations, 0  # Invalid: no score
    nc = _count_colors(colors)
    return 0, nc

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # Safety margin before 90s kill
    rng = np.random.default_rng(42)
    
    best_colors = None
    best_score = -1
    best_violations = None
    best_num_colors = -1
    
    def update_best(colors, violations, nc):
        nonlocal best_colors, best_score, best_violations, best_num_colors
        if violations == 0:
            if nc > best_num_colors:
                best_num_colors = nc
                best_colors = colors.copy()
                best_score = nc
        else:
            if best_violations is None or violations < best_violations or (violations == best_violations and nc > best_num_colors):
                best_violations = violations
                best_num_colors = nc
                best_colors = colors.copy()
                best_score = -violations  # Negative score for invalid
    
    # Initialize with a random valid-ish coloring
    # Start with few colors to avoid violations, then expand
    colors = np.zeros(E, dtype=np.int32)
    
    # Phase 1: Start with a simple valid coloring (all same color)
    # Then gradually introduce more colors
    
    # Use simulated annealing
    # Energy = violations * BIG_PENALTY - num_colors * REWARD
    # We want to minimize energy
    
    BIG_PENALTY = 10000
    REWARD = 1
    
    # Initial state: all zeros (valid, 1 color)
    cur_colors = np.zeros(E, dtype=np.int32)
    cur_violations = 0
    cur_nc = 1
    update_best(cur_colors, cur_violations, cur_nc)
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.01
    num_iterations = 500000  # Will be limited by time
    
    # Precompute: for each edge, list of cycles it's in
    # When we change edge e from old_color to new_color, we need to recheck only cycles containing e
    
    # For faster evaluation, track violation count incrementally
    # violations[cyc] = number of distinct colors in cycle cyc (1 to 4)
    # A cycle is a violation iff violations[cyc] == 4
    
    cyc_distinct = np.zeros(_NUM_CYCLES, dtype=np.int32)
    for ci, cyc in enumerate(_CYCLES):
        c0, c1, c2, c3 = cur_colors[cyc[0]], cur_colors[cyc[1]], cur_colors[cyc[2]], cur_colors[cyc[3]]
        cyc_distinct[ci] = len({c0, c1, c2, c3})
    cur_viol_count = int(np.sum(cyc_distinct == 4))
    
    color_counts = np.zeros(120, dtype=np.int32)
    for c in cur_colors:
        color_counts[c] += 1
    cur_nc = int(np.sum(color_counts > 0))
    
    best_viol_count = cur_viol_count
    best_nc = cur_nc
    best_colors_arr = cur_colors.copy()
    
    # Annealing loop
    for it in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        progress = it / num_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random edge to change
        e = rng.integers(0, E)
        old_color = cur_colors[e]
        
        # Pick a new color: either an existing color or a new one
        # With some probability, try a new color
        if rng.random() < 0.3:
            # Try a new color
            new_color = int(rng.integers(0, 120))
        else:
            # Try an existing color
            existing = np.where(color_counts > 0)[0]
            if len(existing) == 0:
                new_color = 0
            else:
                new_color = int(existing[rng.integers(0, len(existing))])
        
        if new_color == old_color:
            continue
        
        # Compute delta in violations
        # Only cycles containing edge e can change
        affected = _EDGE_TO_CYCLES[e]
        
        delta_viol = 0
        for ci in affected:
            cyc = _CYCLES[ci]
            old_distinct = cyc_distinct[ci]
            
            # Compute new distinct count
            # Get the 4 colors in the cycle
            cs = [cur_colors[cyc[k]] for k in range(4)]
            # Replace the color of edge e
            for k in range(4):
                if cyc[k] == e:
                    cs[k] = new_color
                    break
            new_distinct = len(set(cs))
            
            if old_distinct == 4 and new_distinct < 4:
                delta_viol -= 1
            elif old_distinct < 4 and new_distinct == 4:
                delta_viol += 1
        
        # Compute delta in num_colors
        delta_nc = 0
        if color_counts[old_color] == 1:
            delta_nc -= 1  # old color disappears
        if color_counts[new_color] == 0:
            delta_nc += 1  # new color appears
        
        # Energy delta
        delta_E = BIG_PENALTY * delta_viol - REWARD * delta_nc
        
        # Accept with probability
        if delta_E <= 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_E / max(T, 1e-10))
        
        if accept:
            # Apply change
            cur_colors[e] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            # Update cycle distinct counts
            for ci in affected:
                cyc = _CYCLES[ci]
                cs = [cur_colors[cyc[k]] for k in range(4)]
                cyc_distinct[ci] = len(set(cs))
            
            cur_viol_count += delta_viol
            cur_nc += delta_nc
            
            # Update best
            if cur_viol_count < best_viol_count or (cur_viol_count == best_viol_count and cur_nc > best_nc):
                best_viol_count = cur_viol_count
                best_nc = cur_nc
                best_colors_arr = cur_colors.copy()
    
    # Final: return best valid coloring if found, else best invalid
    if best_viol_count == 0:
        result = best_colors_arr.tolist()
    else:
        # Return the best we found (will be scored as invalid)
        result = best_colors_arr.tolist()
    
    # Ensure all colors are in 0..119
    for i in range(len(result)):
        if result[i] < 0 or result[i] > 119:
            result[i] = 0
    
    return result
# EVOLVE-BLOCK-END
