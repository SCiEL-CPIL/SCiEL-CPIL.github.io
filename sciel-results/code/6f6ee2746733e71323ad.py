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

def _local_search(colors, max_time=45.0):
    """
    Local search: try swapping colors of edges to eliminate rainbow 4-cycles.
    Returns (best_colors, best_score) where score = -violations + used_colors * 100.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    best_colors = colors[:]
    best_viol = _count_violations(best_colors)
    best_used = len(set(best_colors))
    best_score = -best_viol + best_used * 100
    
    random.seed(42)
    
    # Precompute all 4-cycles for efficiency
    all_quads = list(combinations(range(N), 4))
    cycle_edges = []
    for quad in all_quads:
        a, b, c, d = quad
        cycle_edges.append([
            [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)],
            [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)],
            [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        ])
    
    def _get_violation_count(colors, cycle_edges):
        v = 0
        for cycs in cycle_edges:
            for cyc in cycs:
                cs = [colors[e] for e in cyc]
                if len(set(cs)) == 4:
                    v += 1
        return v
    
    # Randomized local search
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        
        # Try assigning a random color
        new_color = random.randint(0, 119)
        if new_color == colors[e_idx]:
            continue
        
        old_color = colors[e_idx]
        colors[e_idx] = new_color
        
        viol = _get_violation_count(colors, cycle_edges)
        used = len(set(colors))
        score = -viol + used * 100
        
        if score > best_score:
            best_score = score
            best_viol = viol
            best_used = used
            best_colors = colors[:]
        else:
            colors[e_idx] = old_color
    
    return best_colors, best_score

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Uses modular arithmetic + local search to maximize colors.
    """
    start_time = time.time()
    
    # Fallback: linear ordering (15 colors, proven valid)
    fallback = _linear_ordering()
    
    # Try various modular constructions
    best_colors = fallback
    best_viol = 0
    best_used = 15
    
    # Try many modular parameters
    mods_to_try = [17, 19, 23, 29, 31, 37]
    ab_combos = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,4), (4,1), (4,3), (5,2), (5,7)]
    
    for mod in mods_to_try:
        for a, b in ab_combos:
            colors = _modular_construction(a, b, mod)
            viol = _count_violations(colors)
            used = len(set(colors))
            
            if viol == 0 and used > best_used:
                best_used = used
                best_colors = colors
                best_viol = 0
            
            # Even if not valid, keep track for local search starting point
            if viol < 10:  # Few violations, good starting point
                # Run local search on this
                deadline_remaining = 45.0 - (time.time() - start_time)
                if deadline_remaining > 5:
                    ls_colors, ls_score = _local_search(colors, max_time=min(deadline_remaining, 10.0))
                    ls_viol = _count_violations(ls_colors)
                    ls_used = len(set(ls_colors))
                    if ls_viol == 0 and ls_used > best_used:
                        best_used = ls_used
                        best_colors = ls_colors
                        best_viol = 0
    
    # Also try local search starting from linear ordering with perturbation
    if time.time() - start_time < 40:
        perturbed = fallback[:]
        random.seed(123)
        for _ in range(20):
            e = random.randint(0, NUM_EDGES - 1)
            perturbed[e] = random.randint(0, 119)
        
        deadline_remaining = 45.0 - (time.time() - start_time)
        if deadline_remaining > 5:
            ls_colors, ls_score = _local_search(perturbed, max_time=min(deadline_remaining, 8.0))
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

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
