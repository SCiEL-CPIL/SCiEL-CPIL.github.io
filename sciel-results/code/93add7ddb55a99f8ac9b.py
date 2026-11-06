import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    # Fallback: n=9 seed
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best = {'n': n_seed, 'colors': seed_colors}

    # Try algebraic construction for n=17
    r17 = _algebraic_construct(17, deadline - 30.0)
    if r17 is not None:
        best = r17

    # Try to extend to n=18
    if time.time() < deadline - 10.0:
        r18 = _extend_to_18(best, deadline - 10.0)
        if r18 is not None:
            return r18

    return best


def _algebraic_construct(n, deadline):
    """Use quadratic residues over Z_19 to construct a C6-free 4-coloring of K_17."""
    if n != 17:
        return None
    
    p = 19  # prime ≡ 3 mod 4
    # Quadratic residues mod 19: {1,4,5,6,7,9,11,16,17}
    residues = [x*x % p for x in range(1, p)]
    residues = sorted(set(residues))
    # Non-residues
    non_residues = sorted([x for x in range(1, p) if x not in residues])
    
    # We have 9 residues and 9 non-residues, plus 0
    # Map each difference to a color in {0,1,2,3}
    # Strategy: partition the 18 nonzero elements into 4 groups of ~4-5 each
    
    # Better: use a known good partition
    # Color 0: residues that are also cubes
    # Let's try a direct mapping
    # Elements: 1..18
    # We need to assign each element d to color c(d) in {0,1,2,3}
    # such that no monochromatic C6 exists
    
    # Known approach: use the fact that in Z_19, the equation
    # x + y + z = 0 with x,y,z in same color class should have no solution
    # that forms a C6
    
    # Let's try: color based on discrete log or simple partition
    # Partition 1..18 into 4 sets:
    # A = {1, 4, 5, 6, 7, 9, 11, 16, 17} (quadratic residues)
    # B = {2, 3, 8, 10, 12, 13, 14, 15, 18} (non-residues)
    # This gives only 2 colors. We need 4.
    
    # Split residues into 2 sets and non-residues into 2 sets
    # Try: color by (discrete log mod 4) or similar
    
    # Let's use a greedy/local search approach with the algebraic structure as initialization
    edge_map, edge_list = _build_edge_map(n)
    num_edges = len(edge_list)
    
    # Initialize with algebraic coloring
    colors = _init_algebraic(n, edge_map)
    
    # Validate
    if _is_valid_fast(n, colors, edge_map):
        return {'n': n, 'colors': colors}
    
    # Local search to fix any violations
    result = _local_search(n, colors, edge_map, deadline)
    if result is not None:
        return result
    
    return None


def _init_algebraic(n, edge_map):
    """Initialize coloring using quadratic residue structure over Z_19."""
    p = 19
    colors = [0] * len(edge_map)
    
    # For each edge (i, j), compute d = (j - i) mod 19
    # Map d to a color based on quadratic character and additional structure
    for (i, j), idx in edge_map.items():
        d = (j - i) % p
        if d == 0:
            colors[idx] = 0
            continue
        # Use a hash-like function to distribute into 4 colors
        # Based on quadratic character and parity of discrete log
        is_res = (d*d % p) in [x*x % p for x in range(1, p)]
        # Actually, d is a residue iff d^((p-1)/2) == 1 mod p
        # Simpler: precompute
        if d in [1,4,5,6,7,9,11,16,17]:
            # Residue: split into 2 colors
            if d in [1, 5, 7, 11, 17]:
                colors[idx] = 0
            else:
                colors[idx] = 1
        else:
            # Non-residue: split into 2 colors
            if d in [2, 8, 10, 13, 18]:
                colors[idx] = 2
            else:
                colors[idx] = 3
    
    return colors


def _is_valid_fast(n, colors, edge_map):
    """Fast validation using adjacency matrices."""
    adj = [_build_adj_matrix(n, colors, edge_map, c) for c in range(4)]
    for c in range(4):
        if _has_c6(adj[c]):
            return False
    return True


def _build_adj_matrix(n, colors, edge_map, color):
    """Build adjacency matrix for a specific color."""
    adj = np.zeros((n, n), dtype=np.int8)
    for (i, j), idx in edge_map.items():
        if colors[idx] == color:
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


def _has_c6(adj):
    """Check if adjacency matrix contains a C6 using matrix multiplication."""
    n = adj.shape[0]
    # A C6 exists iff there's a cycle of length 6
    # Use: (A^6) has diagonal entries > 0 for some vertex
    # But this counts all closed walks of length 6, not just simple cycles
    # Better: check for 6 distinct vertices v1..v6 with edges v1-v2, v2-v3, ..., v6-v1
    
    # Efficient: for each pair of opposite vertices (v1, v4), check if there's
    # a path of length 3 from v1 to v4 that avoids v1 and v4 internally
    # Actually, let's use a simpler but effective method
    
    # For n=17, we can do this efficiently
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                continue
            # Find common neighbors of i and j
            common_ij = np.where((adj[i] & adj[j]) > 0)[0]
            common_ij = common_ij[(common_ij != i) & (common_ij != j)]
            if len(common_ij) < 2:
                continue
            # For each pair of common neighbors (a, b), check if there's
            # a path of length 2 from a to b avoiding i, j
            for a in common_ij:
                for b in common_ij:
                    if a == b or a > b:
                        continue
                    # Check if a and b have a common neighbor c != a, b, i, j
                    # such that c is connected to both a and b
                    common_ab = np.where((adj[a] & adj[b]) > 0)[0]
                    common_ab = common_ab[(common_ab != a) & (common_ab != b) & 
                                         (common_ab != i) & (common_ab != j)]
                    if len(common_ab) >= 1:
                        return True
    return False


def _local_search(n, colors, edge_map, deadline):
    """Local search to eliminate C6 violations."""
    random.seed(42)
    num_edges = len(edge_map)
    
    # Precompute which cycles each edge belongs to
    cycles, edge_to_cycles = _build_cycles_fast(n, edge_map)
    num_cycles = len(cycles)
    
    if num_cycles == 0:
        return {'n': n, 'colors': colors}
    
    # Count violations
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_viol = 0
    for c_id in range(num_cycles):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n, 'colors': colors}
        
        e = random.randint(0, num_edges - 1)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
    return None


def _extend_to_18(best, deadline):
    """Try to extend a valid coloring to n=18."""
    if best is None or best['n'] != 17:
        return None
    
    n_new = 18
    old_colors = best['colors']
    
    edge_map, edge_list = _build_edge_map(n_new)
    num_edges = len(edge_list)
    n17_edges = 17 * 16 // 2
    
    colors = list(old_colors)
    random.seed(42)
    for i in range(n17_edges, num_edges):
        colors[i] = random.randint(0, 3)
    
    # Quick validation
    if _is_valid_fast(n_new, colors, edge_map):
        return {'n': n_new, 'colors': colors}
    
    # Local search on new edges only
    cycles, edge_to_cycles = _build_cycles_fast(n_new, edge_map)
    num_cycles = len(cycles)
    
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_viol = 0
    for c_id in range(num_cycles):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    if total_viol == 0:
        return {'n': n_new, 'colors': colors}
    
    new_edge_indices = list(range(n17_edges, num_edges))
    
    T = 1.0
    cooling = 0.999
    min_T = 0.02
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n_new, 'colors': colors}
        
        e = random.choice(new_edge_indices)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n_new, 'colors': colors}
    return None


def _build_edge_map(n):
    edge_map = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    return edge_map, edge_list


def _build_cycles_fast(n, edge_map):
    """Build cycles using combinations only (no permutations)."""
    cycles = []
    edge_to_cycles = {e: [] for e in range(n * (n - 1) // 2)}
    
    for subset in itertools.combinations(range(n), 6):
        # Generate unique 6-cycles on these 6 vertices
        # Fix vertex 0 as start to avoid rotation duplicates
        v0 = subset[0]
        remaining = list(subset[1:])
        
        for perm in itertools.permutations(remaining):
            # Cycle: v0, perm[0], perm[1], ..., perm[4], v0
            # To avoid reversal duplicates, require perm[0] < perm[4]
            if perm[0] > perm[4]:
                continue
            
            rot = (v0,) + perm
            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])
            
            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)
    
    return cycles, edge_to_cycles
# EVOLVE-BLOCK-END
