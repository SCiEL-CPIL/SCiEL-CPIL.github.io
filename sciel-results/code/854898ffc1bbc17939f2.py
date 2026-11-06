"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _c6_count_for_color(adj_mask, n):
    """Count C6 cycles in a single color class given as a list of bitmasks."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                # Find common neighbors of i and j
                common = adj_mask[i] & adj_mask[j]
                # For each pair of common neighbors (a, b) with a < b,
                # check if a and b are connected (forming a 4-cycle i-j-b-a-i... wait)
                # Actually for C6: i - j - a - b - c - d - i
                # We need to count 6-cycles. A 6-cycle is determined by 6 vertices.
                # Let's use a different approach: count all 6-cycles by checking
                # all combinations of 6 vertices? That's O(n^6) which is too slow for n=17.
                # 
                # Better: A C6 can be represented as a sequence v0-v1-v2-v3-v4-v5-v0.
                # Each C6 is counted 12 times (6 starting vertices * 2 directions).
                # 
                # Alternative: use the fact that a C6 in a graph G is a set of 6 vertices
                # that form a cycle. We can count them by:
                # For each pair of non-adjacent vertices (u, v), count paths of length 5 between them
                # where all intermediate vertices are distinct and not u or v.
                # This is complex.
                #
                # Simpler approach for small n: enumerate all 6-subsets and check if they contain a C6.
                # But that's C(17,6) = 12376 per color, times 4 colors = ~50k checks.
                # Each check is O(1) if we precompute. Actually checking if a 6-set contains a C6
                # requires checking all 6! / (6*2) = 60 possible orderings. That's 50k * 60 = 3M ops.
                # That's feasible.
                pass
    return count

def _has_c6_in_color(adj_mask, n):
    """Check if a single color class contains a C6. adj_mask[i] is a bitmask of neighbors of i in this color."""
    # Enumerate all 6-subsets
    from itertools import combinations
    for combo in combinations(range(n), 6):
        # Check if these 6 vertices contain a C6 in this color
        # A C6 is a cycle of length 6 using all 6 vertices
        # There are (6-1)! / 2 = 60 distinct cycles on 6 labeled vertices
        # But we only need to check if ANY 6-cycle exists using exactly these 6 vertices
        # A 6-cycle on vertices {v0,...,v5} exists if there's a permutation such that
        # consecutive vertices (and last-first) are all edges in this color.
        # 
        # More efficient: check if the induced subgraph on these 6 vertices contains a Hamiltonian cycle.
        # For 6 vertices, we can do this by checking all 60 possible cycles.
        vs = combo
        # Try all permutations of the 6 vertices, but fix one to avoid rotation duplicates
        # Actually, just check: does there exist a cyclic ordering of the 6 vertices
        # such that all 6 consecutive pairs (including wrap-around) are edges?
        # 
        # We can use a simple backtracking or just check all 60 cycles.
        # Let's precompute the adjacency within this 6-set
        # adj[i][j] for i,j in range(6)
        adj6 = [[False]*6 for _ in range(6)]
        for a in range(6):
            for b in range(a+1, 6):
                u, v = vs[a], vs[b]
                if adj_mask[u] & (1 << v):
                    adj6[a][b] = True
                    adj6[b][a] = True
        
        # Now check if adj6 contains a Hamiltonian cycle
        # Fix vertex 0 as start to avoid rotation duplicates
        # Try all permutations of vertices 1-5
        from itertools import permutations
        for perm in permutations(range(1, 6)):
            # Cycle: 0, perm[0], perm[1], perm[2], perm[3], perm[4], 0
            ok = True
            for k in range(5):
                if not adj6[perm[k]][perm[k+1]]:
                    ok = False
                    break
            if ok and not adj6[perm[4]][0]:
                ok = False
            if ok:
                return True
    return False

def _validate(colors, n):
    """Validate that no color class contains a C6."""
    # Build adjacency masks for each color
    adj_masks = [[0]*n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj_masks[c][i] |= (1 << j)
            adj_masks[c][j] |= (1 << i)
            idx += 1
    for c in range(4):
        if _has_c6_in_color(adj_masks[c], n):
            return False
    return True

def _random_search(n, timeout=10):
    """Random search for a valid 4-coloring of K_n."""
    start = time.time()
    num_edges = n * (n - 1) // 2
    while time.time() - start < timeout:
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        if _validate(colors, n):
            return colors
    return None

def _improve(colors, n, timeout=5):
    """Try to improve a near-valid coloring by flipping individual edges."""
    start = time.time()
    colors = list(colors)
    num_edges = n * (n - 1) // 2
    
    # First, count C6 violations per color
    def count_violations(col_list):
        adj_masks = [[0]*n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = col_list[idx]
                adj_masks[c][i] |= (1 << j)
                adj_masks[c][j] |= (1 << i)
                idx += 1
        total = 0
        for c in range(4):
            if _has_c6_in_color(adj_masks[c], n):
                total += 1
        return total
    
    current_violations = count_violations(colors)
    if current_violations == 0:
        return colors
    
    while time.time() - start < timeout:
        # Pick a random edge to flip
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.choice([c for c in range(4) if c != old_color])
        colors[e] = new_color
        
        new_violations = count_violations(colors)
        if new_violations < current_violations:
            current_violations = new_violations
            if current_violations == 0:
                return colors
        else:
            # Revert with some probability (simulated annealing style)
            if random.random() > 0.1:
                colors[e] = old_color
    
    return colors if count_violations(colors) == 0 else None

def construct():
    """Find a 4-coloring of K_n with no monochromatic C6, maximizing n."""
    total_start = time.time()
    deadline = total_start + 130  # Leave 20s margin
    
    # Start with known valid colorings and try to extend
    # The seed gives n=9. Let's try to find larger ones.
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try random search for n=10 through n=15
    for n in range(10, 16):
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on n (smaller n gets more time per attempt)
        alloc = min(10, remaining / (16 - n))
        
        # Try random search
        result = _random_search(n, timeout=alloc)
        if result is not None:
            best_n = n
            best_colors = result
            continue
        
        # Try starting from a valid coloring of smaller n and extending
        # This is complex, so skip for now
    
    # Try to improve with local search if we have time
    if best_n >= 10 and time.time() < deadline - 10:
        improved = _improve(best_colors, best_n, timeout=min(10, deadline - time.time() - 5))
        if improved is not None:
            best_colors = improved
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
