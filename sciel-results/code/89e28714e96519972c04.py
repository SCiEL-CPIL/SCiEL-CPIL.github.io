"""
Edge-coloring of K_16 to maximize distinct colors with no rainbow C4.
Combines structured baseline + efficient local search from current program
with simulated-annealing-style exploration and numpy-based validation from inspiration.
"""
import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# --- Precomputation (done once at module load) ---

def _build_edges():
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1
    return edges, edge_idx

EDGES, EDGE_IDX = _build_edges()

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((EDGE_IDX[(a,b)], EDGE_IDX[(b,c)], EDGE_IDX[(c,d)], EDGE_IDX[(a,d)]))
                    # Cycle 2: a-b, b-d, c-d, a-c
                    cycles.append((EDGE_IDX[(a,b)], EDGE_IDX[(b,d)], EDGE_IDX[(c,d)], EDGE_IDX[(a,c)]))
                    # Cycle 3: a-c, b-c, b-d, a-d
                    cycles.append((EDGE_IDX[(a,c)], EDGE_IDX[(b,c)], EDGE_IDX[(b,d)], EDGE_IDX[(a,d)]))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# For each edge, list of cycle indices it belongs to
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)

# For fast numpy validation: precompute cycles as a 2D array
CYCLES_ARR = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def _count_violations_np(colors_np):
    """Count rainbow 4-cycles using numpy vectorization."""
    # colors_np: shape (120,)
    # For each cycle, get the 4 colors
    c = colors_np[CYCLES_ARR]  # shape (NUM_CYCLES, 4)
    # A cycle is rainbow iff all 4 colors are distinct
    # Check: max != min OR use set logic. Fast way: 
    # Sort each row and check no adjacent equals
    c_sorted = np.sort(c, axis=1)
    # rainbow iff c_sorted[:,0] != c_sorted[:,1] and c_sorted[:,1] != c_sorted[:,2] and c_sorted[:,2] != c_sorted[:,3]
    mask = (c_sorted[:, 0] != c_sorted[:, 1]) & \
           (c_sorted[:, 1] != c_sorted[:, 2]) & \
           (c_sorted[:, 2] != c_sorted[:, 3])
    return int(np.sum(mask))

def _count_violations(colors):
    """Count rainbow 4-cycles (list-based)."""
    count = 0
    for cycle in CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            count += 1
    return count

def _count_distinct(colors):
    return len(set(colors))

def _check_edge_change(colors, edge_i, new_color):
    """Check if changing edge_i to new_color creates/removes violations.
    Returns (num_new, num_removed)."""
    old_color = colors[edge_i]
    if old_color == new_color:
        return 0, 0
    
    num_new = 0
    num_removed = 0
    for ci in EDGE_TO_CYCLES[edge_i]:
        cycle = CYCLES[ci]
        others = [e for e in cycle if e != edge_i]
        o0, o1, o2 = others
        c0, c1, c2 = colors[o0], colors[o1], colors[o2]
        
        # Are the other three already all distinct?
        others_distinct = (c0 != c1 and c0 != c2 and c1 != c2)
        if not others_distinct:
            continue  # Can't be a violation regardless of this edge's color
        
        was_violation = (old_color != c0 and old_color != c1 and old_color != c2)
        is_violation = (new_color != c0 and new_color != c1 and new_color != c2)
        
        if is_violation and not was_violation:
            num_new += 1
        elif was_violation and not is_violation:
            num_removed += 1
    
    return num_new, num_removed

def _make_structured_baseline():
    """
    Structured coloring based on 4x4 grid coordinates.
    Edge (i,j) color = f(coords(i), coords(j)).
    This gives a valid coloring with ~16-25 distinct colors.
    """
    colors = [0] * NUM_EDGES
    vertex_coords = [(i // 4, i % 4) for i in range(16)]
    
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = vertex_coords[i]
            xj, yj = vertex_coords[j]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            # Color based on (dx, dy) — at most 4*4=16 values
            colors[EDGE_IDX[(i, j)]] = dx * 4 + dy
    return colors

def _make_valid_random_baseline(rng):
    """
    Build a valid coloring by greedy assignment with backtracking-lite.
    Start with all zeros, then try to introduce new colors one by one.
    """
    colors = [0] * NUM_EDGES
    used = {0}
    
    # Try to add new colors to random edges
    for _ in range(300):
        ei = rng.randint(0, NUM_EDGES - 1)
        old = colors[ei]
        # Try a new color
        nc = rng.randint(0, MAX_COLOR - 1)
        if nc == old:
            continue
        num_new, num_removed = _check_edge_change(colors, ei, nc)
        if num_new == 0:
            colors[ei] = nc
            used.add(nc)
    
    # Verify
    if _count_violations(colors) == 0:
        return colors
    return None

def _local_search_maximize_colors(colors, deadline, rng):
    """
    Local search to maximize distinct colors while maintaining validity.
    Uses a multi-phase approach:
    Phase 1: Introduce new colors (greedy)
    Phase 2: Swap colors to free up slots for new ones
    Phase 3: Simulated annealing for broader exploration
    """
    colors = list(colors)
    used_colors = set(colors)
    best_colors = list(colors)
    best_distinct = len(used_colors)
    
    t0 = time.time()
    
    # Phase 1: Greedy new-color introduction
    max_phase1 = 20000
    for it in range(max_phase1):
        if time.time() - t0 > deadline:
            break
        
        improved = False
        # Try to change a random edge to a completely new color
        for _ in range(15):
            ei = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[ei]
            
            # Candidate new colors: prefer ones not yet used
            # Sample a few candidates
            candidates = []
            for _ in range(8):
                c = rng.randint(0, MAX_COLOR - 1)
                if c not in used_colors:
                    candidates.append(c)
            
            if not candidates:
                break
            
            best_nc = None
            for nc in candidates:
                num_new, num_removed = _check_edge_change(colors, ei, nc)
                if num_new == 0:
                    best_nc = nc
                    break
            
            if best_nc is not None:
                colors[ei] = best_nc
                used_colors.add(best_nc)
                improved = True
                if len(used_colors) > best_distinct:
                    best_distinct = len(used_colors)
                    best_colors = list(colors)
                break
        
        if not improved:
            # Try changing to any color (including used) that doesn't create violations,
            # then try to free up a slot
            any_change = False
            for _ in range(10):
                ei = rng.randint(0, NUM_EDGES - 1)
                old_color = colors[ei]
                nc = rng.randint(0, MAX_COLOR - 1)
                if nc == old_color:
                    continue
                num_new, _ = _check_edge_change(colors, ei, nc)
                if num_new == 0:
                    colors[ei] = nc
                    any_change = True
                    break
            if not any_change:
                break
    
    # Phase 2: Simulated annealing to explore color space
    # Objective: maximize distinct colors, minimize violations (should stay 0)
    T_start = 5.0
    T_end = 0.1
    total_sa_steps = 30000
    sa_step = 0
    
    # Track color counts for entropy bonus
    def color_entropy_bonus(colors_list):
        from collections import Counter
        cnt = Counter(colors_list)
        n = len(colors_list)
        if not cnt:
            return 0.0
        probs = [c / n for c in cnt.values()]
        ent = -sum(p * np.log2(p) for p in probs)
        return ent
    
    current_entropy = color_entropy_bonus(colors)
    
    while sa_step < total_sa_steps:
        if time.time() - t0 > deadline:
            break
        
        sa_step += 1
        T = T_start * (T_end / T_start) ** (sa_step / total_sa_steps)
        
        # Pick a random edge
        ei = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Pick new color: biased toward new colors
        if rng.random() < 0.4:
            nc = rng.randint(0, MAX_COLOR - 1)
        else:
            # Pick from colors in neighboring cycles or random
            neighbors = set()
            for ci in EDGE_TO_CYCLES[ei]:
                for e in CYCLES[ci]:
                    if e != ei:
                        neighbors.add(colors[e])
            if neighbors:
                nc = rng.choice(list(neighbors))
            else:
                nc = rng.randint(0, MAX_COLOR - 1)
        
        if nc == old_color:
            continue
        
        num_new, num_removed = _check_edge_change(colors, ei, nc)
        
        # Delta in "energy": we want 0 violations and max distinct colors
        # Simplified: accept if no new violations, or with probability exp(-delta/T) if violations increase
        if num_new == 0:
            # Accept if it adds a new color or maintains
            old_distinct = len(used_colors)
            was_unused = nc not in used_colors
            colors[ei] = nc
            used_colors = set(colors)
            new_distinct = len(used_colors)
            
            # Update best
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = list(colors)
        else:
            # Reject (we need to stay valid)
            pass
    
    # Final: try one more greedy pass
    for it in range(5000):
        if time.time() - t0 > deadline:
            break
        improved = False
        for _ in range(10):
            ei = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[ei]
            nc = rng.randint(0, MAX_COLOR - 1)
            if nc in used_colors:
                continue
            num_new, _ = _check_edge_change(colors, ei, nc)
            if num_new == 0:
                colors[ei] = nc
                used_colors.add(nc)
                improved = True
                if len(used_colors) > best_distinct:
                    best_distinct = len(used_colors)
                    best_colors = list(colors)
                break
        if not improved:
            break
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    rng = random.Random(12345)
    
    best_colors = None
    best_distinct = 0
    
    # Strategy 1: Structured baseline + local search
    base1 = _make_structured_baseline()
    v1 = _count_violations(base1)
    if v1 == 0:
        improved, d1 = _local_search_maximize_colors(base1, DEADLINE - (time.time() - start_time), rng)
        if d1 > best_distinct:
            best_distinct = d1
            best_colors = improved
    
    # Strategy 2: Random valid baseline + local search
    if time.time() - start_time < DEADLINE - 10:
        base2 = _make_valid_random_baseline(rng)
        if base2 is not None:
            improved2, d2 = _local_search_maximize_colors(base2, DEADLINE - (time.time() - start_time), rng)
            if d2 > best_distinct:
                best_distinct = d2
                best_colors = improved2
    
    # Strategy 3: Multiple random restarts with local search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        num_restarts = min(20, int(remaining / 3))
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE - 2:
                break
            
            # Random initial coloring
            colors = [rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)]
            
            # Quick violation reduction
            for _ in range(5000):
                if time.time() - start_time > DEADLINE - 2:
                    break
                violations = _count_violations(colors)
                if violations == 0:
                    break
                
                # Pick an edge in a violating cycle
                changed = False
                for _ in range(10):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    # Check if in violation
                    in_viol = False
                    for ci in EDGE_TO_CYCLES[ei]:
                        cycle = CYCLES[ci]
                        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                            in_viol = True
                            break
                    if not in_viol:
                        continue
                    
                    # Try to fix: pick a color from neighbors
                    candidates = set()
                    for ci in EDGE_TO_CYCLES[ei]:
                        for e in CYCLES[ci]:
                            if e != ei:
                                candidates.add(colors[e])
                    
                    best_nc = None
                    for nc in candidates:
                        if nc == colors[ei]:
                            continue
                        num_new, num_removed = _check_edge_change(colors, ei, nc)
                        if num_removed > num_new:
                            best_nc = nc
                            break
                        if num_new == 0 and num_removed > 0:
                            best_nc = nc
                            break
                    
                    if best_nc is None:
                        # Try any color that reduces violations
                        for nc in candidates:
                            if nc == colors[ei]:
                                continue
                            num_new, num_removed = _check_edge_change(colors, ei, nc)
                            if num_removed > num_new:
                                best_nc = nc
                                break
                    
                    if best_nc is not None:
                        colors[ei] = best_nc
                        changed = True
                        break
                
                if not changed:
                    # Perturb
                    ei = rng.randint(0, NUM_EDGES - 1)
                    colors[ei] = rng.randint(0, MAX_COLOR - 1)
            
            # Check if valid
            if _count_violations(colors) == 0:
                # Now maximize colors
                rem = DEADLINE - (time.time() - start_time)
                if rem > 2:
                    improved, d = _local_search_maximize_colors(colors, rem, rng)
                    if d > best_distinct:
                        best_distinct = d
                        best_colors = improved
    
    # Final validation
    if best_colors is None:
        # Fallback: all zeros
        return [0] * NUM_EDGES
    
    final_v = _count_violations(best_colors)
    if final_v > 0:
        # Shouldn't happen, but fallback
        colors2 = [0] * NUM_EDGES
        return colors2
    
    return best_colors


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    print(f"Time: {elapsed:.2f}s")
    # Validate
    v = _count_violations(result)
    print(f"Violations: {v}")
