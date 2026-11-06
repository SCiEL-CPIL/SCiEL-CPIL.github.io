import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _count_violations(colors):
    """Count the number of rainbow 4-cycles."""
    violations = 0
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        c1 = [colors[edge_idx(a,b)], colors[edge_idx(b,c)], colors[edge_idx(c,d)], colors[edge_idx(d,a)]]
        if len(set(c1)) == 4:
            violations += 1
        c2 = [colors[edge_idx(a,b)], colors[edge_idx(b,d)], colors[edge_idx(d,c)], colors[edge_idx(c,a)]]
        if len(set(c2)) == 4:
            violations += 1
        c3 = [colors[edge_idx(a,c)], colors[edge_idx(c,b)], colors[edge_idx(b,d)], colors[edge_idx(d,a)]]
        if len(set(c3)) == 4:
            violations += 1
    return violations

def _is_valid(colors):
    return _count_violations(colors) == 0

def _linear_ordering():
    """Proven valid coloring with N-1 = 15 colors."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = j
    return colors

def _modular_construction(a, b, mod):
    """Generate coloring c(i,j) = (a*i + b*j) % mod."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = (a * i + b * j) % mod
    return colors

def _build_cycle_structure():
    """Precompute all 4-cycles and edge-to-cycle incidence for incremental updates."""
    all_quads = list(combinations(range(N), 4))
    cycles = []  # list of lists of 4 edge indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in all_quads:
        a, b, c, d = quad
        # Three 4-cycles on vertices {a,b,c,d}
        c1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        c2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        c3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        
        for cyc in [c1, c2, c3]:
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles) - 1)
    
    return cycles, edge_to_cycles

def _is_rainbow(cycle, colors):
    """Check if a 4-cycle has all distinct colors."""
    c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
    return len({c0, c1, c2, c3}) == 4

def _local_search(colors, max_time=45.0):
    """
    Local search with incremental violation counting.
    Uses a greedy strategy: prioritize edges that can introduce new colors
    without creating rainbow cycles.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    cycles, edge_to_cycles = _build_cycle_structure()
    
    # Track which cycles are currently rainbow
    rainbow_flags = [False] * len(cycles)
    viol_count = 0
    for i, cyc in enumerate(cycles):
        if _is_rainbow(cyc, colors):
            rainbow_flags[i] = True
            viol_count += 1
    
    used_colors = set(colors)
    best_colors = colors[:]
    best_viol = viol_count
    best_used = len(used_colors)
    best_score = -best_viol + best_used * 100
    
    random.seed(42)
    
    # Precompute for each edge, the set of colors that appear on its incident cycles
    # This helps us know which new colors are "safe"
    
    def try_assign(e_idx, new_color):
        """Try assigning new_color to edge e_idx. Returns (delta_viol, delta_used) or None if no change."""
        old_color = colors[e_idx]
        if new_color == old_color:
            return None
        
        # Compute change in violation count
        delta_viol = 0
        for ci in edge_to_cycles[e_idx]:
            was_rainbow = rainbow_flags[ci]
            # Temporarily set new color
            colors[e_idx] = new_color
            now_rainbow = _is_rainbow(cycles[ci], colors)
            # Restore
            colors[e_idx] = old_color
            
            if was_rainbow and not now_rainbow:
                delta_viol -= 1
            elif not was_rainbow and now_rainbow:
                delta_viol += 1
        
        # Compute change in used colors
        old_used = len(used_colors)
        # Simulate
        if new_color not in used_colors:
            # Check if old_color still appears elsewhere
            # For efficiency, just compute
            pass
        # Simplified: just count
        new_used_set = set(colors)
        new_used_set.discard(old_color)
        new_used_set.add(new_color)
        delta_used = len(new_used_set) - old_used
        
        return delta_viol, delta_used
    
    # Greedy + random hybrid local search
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Strategy 1: Try to introduce a new color on a random edge
        # Strategy 2: Try to reduce violations by changing an edge in a rainbow cycle
        # Strategy 3: Random perturbation
        
        mode = random.random()
        
        if mode < 0.4:
            # Try to introduce new color
            e_idx = random.randint(0, NUM_EDGES - 1)
            # Choose a color not currently used
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                continue
            new_color = random.choice(unused)
        elif mode < 0.7:
            # Try to fix a violation
            if viol_count == 0:
                mode = 0.99  # fall through to random
            else:
                # Pick a random rainbow cycle
                rainbow_cycles = [i for i in range(len(cycles)) if rainbow_flags[i]]
                if not rainbow_cycles:
                    continue
                ci = random.choice(rainbow_cycles)
                cyc = cycles[ci]
                e_idx = random.choice(cyc)
                # Try a color that appears elsewhere in the cycle to break rainbow
                other_colors = [colors[e] for e in cyc if e != e_idx]
                if other_colors:
                    new_color = random.choice(other_colors)
                else:
                    new_color = random.randint(0, 119)
        else:
            # Random edge, random color
            e_idx = random.randint(0, NUM_EDGES - 1)
            new_color = random.randint(0, 119)
        
        result = try_assign(e_idx, new_color)
        if result is None:
            continue
        
        delta_viol, delta_used = result
        new_score = best_score + (-delta_viol) + (delta_used * 100)
        
        # Accept if score improves, or with some probability if neutral (for exploration)
        if new_score > best_score or (new_score == best_score and random.random() < 0.1):
            # Apply the change
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            
            # Update rainbow flags
            for ci in edge_to_cycles[e_idx]:
                was_rainbow = rainbow_flags[ci]
                now_rainbow = _is_rainbow(cycles[ci], colors)
                if was_rainbow != now_rainbow:
                    rainbow_flags[ci] = now_rainbow
                    viol_count += 1 if now_rainbow else -1
            
            # Update used colors
            used_colors.discard(old_color)
            used_colors.add(new_color)
            
            best_viol = viol_count
            best_used = len(used_colors)
            best_score = -best_viol + best_used * 100
            best_colors = colors[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_colors, best_score

def _greedy_color_introduction(colors, max_time=30.0):
    """
    Greedy strategy: iteratively select the edge that maximizes distinct colors
    in its 4-cycle neighborhood without creating a rainbow cycle.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    cycles, edge_to_cycles = _build_cycle_structure()
    
    # Track rainbow status
    rainbow_flags = [False] * len(cycles)
    viol_count = 0
    for i, cyc in enumerate(cycles):
        if _is_rainbow(cyc, colors):
            rainbow_flags[i] = True
            viol_count += 1
    
    used_colors = set(colors)
    best_colors = colors[:]
    best_used = len(used_colors)
    best_viol = viol_count
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        if best_viol > 0:
            break  # Can't safely introduce more colors if we have violations
        
        best_gain = 0
        best_e = -1
        best_c = -1
        
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            for c in range(120):
                if c in used_colors:
                    continue
                # Check if assigning c to e creates any rainbow cycle
                old_c = colors[e]
                safe = True
                for ci in edge_to_cycles[e]:
                    if not rainbow_flags[ci]:
                        # Check if it would become rainbow
                        colors[e] = c
                        if _is_rainbow(cycles[ci], colors):
                            colors[e] = old_c
                            safe = False
                            break
                        colors[e] = old_c
                if safe:
                    # Count how many new "color diversity" this adds
                    # Simple heuristic: just accept the first valid one
                    best_gain = 1
                    best_e = e
                    best_c = c
                    break
            if best_e != -1:
                break
        
        if best_e == -1:
            break
        
        # Apply
        old_c = colors[best_e]
        colors[best_e] = best_c
        used_colors.discard(old_c)
        used_colors.add(best_c)
        
        # Update rainbow flags
        for ci in edge_to_cycles[best_e]:
            was_rainbow = rainbow_flags[ci]
            now_rainbow = _is_rainbow(cycles[ci], colors)
            if was_rainbow != now_rainbow:
                rainbow_flags[ci] = now_rainbow
                viol_count += 1 if now_rainbow else -1
        
        best_viol = viol_count
        best_used = len(used_colors)
        best_colors = colors[:]
        
        if best_viol > 0:
            # Revert
            colors[best_e] = old_c
            used_colors.discard(best_c)
            used_colors.add(old_c)
            for ci in edge_to_cycles[best_e]:
                rainbow_flags[ci] = _is_rainbow(cycles[ci], colors)
                viol_count = sum(1 for f in rainbow_flags if f)
            break
    
    return best_colors

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Uses modular arithmetic + local search with incremental updates to maximize colors.
    """
    start_time = time.time()
    HARD_DEADLINE = 85.0  # Leave 5s safety margin
    
    # Fallback: linear ordering (15 colors, proven valid)
    fallback = _linear_ordering()
    
    best_colors = fallback
    best_viol = 0
    best_used = 15
    
    # Try various modular constructions
    mods_to_try = [17, 19, 23, 29, 31, 37, 41, 43]
    ab_combos = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,4), (4,1), (4,3), (5,2), (5,7), (2,5), (3,5), (4,5), (1,4), (2,4), (3,3)]
    
    for mod in mods_to_try:
        for a, b in ab_combos:
            if time.time() > HARD_DEADLINE - 15:
                break
            colors = _modular_construction(a, b, mod)
            viol = _count_violations(colors)
            used = len(set(colors))
            
            if viol == 0 and used > best_used:
                best_used = used
                best_colors = colors[:]
                best_viol = 0
            
            # Even if not valid, keep track for local search starting point
            if viol <= 5:  # Few violations, good starting point
                deadline_remaining = HARD_DEADLINE - 10 - (time.time() - start_time)
                if deadline_remaining > 5:
                    ls_colors, ls_score = _local_search(colors[:], max_time=min(deadline_remaining, 12.0))
                    ls_viol = _count_violations(ls_colors)
                    ls_used = len(set(ls_colors))
                    if ls_viol == 0 and ls_used > best_used:
                        best_used = ls_used
                        best_colors = ls_colors
                        best_viol = 0
    
    # Try greedy color introduction on the best valid coloring found
    if time.time() < HARD_DEADLINE - 10:
        greedy_colors = _greedy_color_introduction(best_colors[:], max_time=8.0)
        greedy_viol = _count_violations(greedy_colors)
        greedy_used = len(set(greedy_colors))
        if greedy_viol == 0 and greedy_used > best_used:
            best_used = greedy_used
            best_colors = greedy_colors
    
    # Also try local search starting from linear ordering with perturbation
    if time.time() < HARD_DEADLINE - 10:
        perturbed = fallback[:]
        random.seed(123)
        for _ in range(30):
            e = random.randint(0, NUM_EDGES - 1)
            perturbed[e] = random.randint(0, 119)
        
        deadline_remaining = HARD_DEADLINE - 10 - (time.time() - start_time)
        if deadline_remaining > 5:
            ls_colors, ls_score = _local_search(perturbed, max_time=min(deadline_remaining, 10.0))
            ls_viol = _count_violations(ls_colors)
            ls_used = len(set(ls_colors))
            if ls_viol == 0 and ls_used > best_used:
                best_used = ls_used
                best_colors = ls_colors
    
    # Final local search polish on the best result
    if time.time() < HARD_DEADLINE - 5:
        deadline_remaining = HARD_DEADLINE - 5 - (time.time() - start_time)
        if deadline_remaining > 3:
            ls_colors, ls_score = _local_search(best_colors[:], max_time=min(deadline_remaining, 8.0))
            ls_viol = _count_violations(ls_colors)
            ls_used = len(set(ls_colors))
            if ls_viol == 0 and ls_used > best_used:
                best_used = ls_used
                best_colors = ls_colors
    
    # Final check
    if _count_violations(best_colors) != 0:
        return fallback
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
