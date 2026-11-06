import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmask list adj on n vertices contains K2,2,2.
    Returns True if found.
    """
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a)
                    cn_mask &= ~(1 << b)
                    cn_mask &= ~(1 << c)
                    cn_mask &= ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        return True
    return False

def _is_valid_coloring(adj, n):
    """Check if neither color class contains K2,2,2."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _build_paley29():
    """Build Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _precompute_pair_props(base_adj, n_base):
    """
    Precompute for every pair (i,j), i<j, whether they can be in the same part
    of a K2,2,2 (i.e., NOT connected in red) and cross-connectivity info.
    Returns a dict mapping (i,j) -> (not_red_edge, red_edge)
    Actually we just need: for each pair, is there a red edge or not.
    """
    # We'll store as a 2D structure for fast lookup
    # pair_red[i][j] = 1 if edge (i,j) is red in base graph
    pair_red = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i+1, n_base):
            if (base_adj[i] >> j) & 1:
                pair_red[i][j] = 1
                pair_red[j][i] = 1
    return pair_red

def _count_violations_fast(x, pair_red, n_base):
    """
    Count K2,2,2 violations involving vertex n_base (the 30th vertex).
    x[i] = 1 means vertex i is red-connected to new vertex, 0 means blue-connected.
    """
    violations = 0
    n = n_base  # vertices 0..n-1 are the base
    
    # Get red and blue neighbor sets as lists
    red_neigh = [i for i in range(n) if x[i] == 1]
    blue_neigh = [i for i in range(n) if x[i] == 0]
    
    # Red violations: new_v is in Part A.
    # Need 4 red neighbors partitioned into {b1,b2} and {c1,c2} where:
    #   b1-b2 NOT red, c1-c2 NOT red, all 4 cross edges ARE red
    if len(red_neigh) >= 4:
        for quad in combinations(red_neigh, 4):
            u, v, w, xt = quad
            # Pairing 1: {u,v} and {w,xt}
            if pair_red[u][v] == 0 and pair_red[w][xt] == 0:
                if pair_red[u][w] and pair_red[u][xt] and pair_red[v][w] and pair_red[v][xt]:
                    violations += 1
            # Pairing 2: {u,w} and {v,xt}
            if pair_red[u][w] == 0 and pair_red[v][xt] == 0:
                if pair_red[u][v] and pair_red[u][xt] and pair_red[w][v] and pair_red[w][xt]:
                    violations += 1
            # Pairing 3: {u,xt} and {v,w}
            if pair_red[u][xt] == 0 and pair_red[v][w] == 0:
                if pair_red[u][v] and pair_red[u][w] and pair_red[xt][v] and pair_red[xt][w]:
                    violations += 1
    
    # Blue violations: new_v is in Part A of blue graph.
    # Blue edge means NOT red edge.
    # Need 4 blue neighbors partitioned into {b1,b2} and {c1,c2} where:
    #   b1-b2 NOT blue (= IS red), c1-c2 NOT blue (= IS red),
    #   all 4 cross edges ARE blue (= NOT red)
    if len(blue_neigh) >= 4:
        for quad in combinations(blue_neigh, 4):
            u, v, w, xt = quad
            # Pairing 1: {u,v} and {w,xt}
            if pair_red[u][v] == 1 and pair_red[w][xt] == 1:
                if pair_red[u][w] == 0 and pair_red[u][xt] == 0 and pair_red[v][w] == 0 and pair_red[v][xt] == 0:
                    violations += 1
            # Pairing 2: {u,w} and {v,xt}
            if pair_red[u][w] == 1 and pair_red[v][xt] == 1:
                if pair_red[u][v] == 0 and pair_red[u][xt] == 0 and pair_red[w][v] == 0 and pair_red[w][xt] == 0:
                    violations += 1
            # Pairing 3: {u,xt} and {v,w}
            if pair_red[u][xt] == 1 and pair_red[v][w] == 1:
                if pair_red[u][v] == 0 and pair_red[u][w] == 0 and pair_red[xt][v] == 0 and pair_red[xt][w] == 0:
                    violations += 1
    
    return violations

def _count_violations_for_vertex_flip(x, pair_red, n_base, flip_idx):
    """
    Count violations that involve flip_idx and the new vertex (n_base).
    This is a subset of all violations - those where flip_idx is one of the 4 neighbors of new_v.
    """
    violations = 0
    n = n_base
    flip_color = x[flip_idx]  # 1=red, 0=blue
    
    if flip_color == 1:
        # flip_idx is red neighbor of new_v
        other_red = [i for i in range(n) if i != flip_idx and x[i] == 1]
        if len(other_red) >= 3:
            for tri in combinations(other_red, 3):
                a, b, c = tri
                # 4 vertices: flip_idx, a, b, c - all red neighbors
                # Need partition into two pairs with no intra-pair red edges, all cross red edges
                # Pairing 1: {flip_idx, a} and {b, c}
                if pair_red[flip_idx][a] == 0 and pair_red[b][c] == 0:
                    if pair_red[flip_idx][b] and pair_red[flip_idx][c] and pair_red[a][b] and pair_red[a][c]:
                        violations += 1
                # Pairing 2: {flip_idx, b} and {a, c}
                if pair_red[flip_idx][b] == 0 and pair_red[a][c] == 0:
                    if pair_red[flip_idx][a] and pair_red[flip_idx][c] and pair_red[b][a] and pair_red[b][c]:
                        violations += 1
                # Pairing 3: {flip_idx, c} and {a, b}
                if pair_red[flip_idx][c] == 0 and pair_red[a][b] == 0:
                    if pair_red[flip_idx][a] and pair_red[flip_idx][b] and pair_red[c][a] and pair_red[c][b]:
                        violations += 1
    else:
        # flip_idx is blue neighbor of new_v
        other_blue = [i for i in range(n) if i != flip_idx and x[i] == 0]
        if len(other_blue) >= 3:
            for tri in combinations(other_blue, 3):
                a, b, c = tri
                # Blue graph: edge means NOT red
                # Pairing 1: {flip_idx, a} and {b, c}
                # Intra-pair must NOT be blue edges => MUST be red edges
                # Cross-pair must be blue edges => must NOT be red edges
                if pair_red[flip_idx][a] == 1 and pair_red[b][c] == 1:
                    if pair_red[flip_idx][b] == 0 and pair_red[flip_idx][c] == 0 and pair_red[a][b] == 0 and pair_red[a][c] == 0:
                        violations += 1
                # Pairing 2: {flip_idx, b} and {a, c}
                if pair_red[flip_idx][b] == 1 and pair_red[a][c] == 1:
                    if pair_red[flip_idx][a] == 0 and pair_red[flip_idx][c] == 0 and pair_red[b][a] == 0 and pair_red[b][c] == 0:
                        violations += 1
                # Pairing 3: {flip_idx, c} and {a, b}
                if pair_red[flip_idx][c] == 1 and pair_red[a][b] == 1:
                    if pair_red[flip_idx][a] == 0 and pair_red[flip_idx][b] == 0 and pair_red[c][a] == 0 and pair_red[c][b] == 0:
                        violations += 1
    
    return violations

def _heuristic_init(base_adj, pair_red, n_base):
    """
    Heuristic initialization: try to balance red/blue connections to new vertex
    while minimizing immediate violations.
    """
    n = n_base
    # Start with balanced assignment
    x = [0] * n
    # Assign roughly half red, half blue
    for i in range(n // 2):
        x[i] = 1
    
    # Greedy improvement: for each vertex, try both colors, keep better
    for _ in range(3):  # a few passes
        for i in range(n):
            v0 = _count_violations_for_vertex_flip(x, pair_red, n_base, i)
            x[i] = 1 - x[i]
            v1 = _count_violations_for_vertex_flip(x, pair_red, n_base, i)
            if v1 > v0:
                x[i] = 1 - x[i]  # revert
    
    return x

def _try_extend_to_30(base_adj, deadline):
    """
    Try to extend the 29-vertex base to 30 vertices using local search minimizing violations.
    Uses precomputed pair properties for fast violation counting.
    """
    n_base = 29
    n_target = 30
    random.seed(42)
    
    pair_red = _precompute_pair_props(base_adj, n_base)
    
    best_x = None
    best_violations = float('inf')
    
    # Precompute neighbor lists for each possible color assignment
    # For speed, we track red_set and blue_set as bitmasks
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart < 5:
            # Use heuristic init for first few restarts
            x = _heuristic_init(base_adj, pair_red, n_base)
        else:
            x = [random.randint(0, 1) for _ in range(n_base)]
        
        v = _count_violations_fast(x, pair_red, n_base)
        
        if v == 0:
            return x[:]
        
        if v < best_violations:
            best_violations = v
            best_x = x[:]
        
        # Local search with simulated annealing
        current_x = x[:]
        current_v = v
        temperature = 2.0
        cooling_rate = 0.998
        
        # Precompute same-color neighbor lists for faster delta computation
        # We'll recompute these as needed
        
        steps_per_restart = 5000
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            i = random.randint(0, n_base - 1)
            
            # Compute delta: violations involving vertex i in old vs new state
            # Old: x[i] = current_x[i]
            # New: x[i] = 1 - current_x[i]
            
            old_color = current_x[i]
            
            # Count violations involving i in old state
            if old_color == 1:
                other_same = [j for j in range(n_base) if j != i and current_x[j] == 1]
            else:
                other_same = [j for j in range(n_base) if j != i and current_x[j] == 0]
            
            old_viol = 0
            if len(other_same) >= 3:
                for tri in combinations(other_same, 3):
                    a, b, c = tri
                    if old_color == 1:
                        # Red: intra-pair NOT red, cross red
                        if pair_red[i][a] == 0 and pair_red[b][c] == 0:
                            if pair_red[i][b] and pair_red[i][c] and pair_red[a][b] and pair_red[a][c]:
                                old_viol += 1
                        if pair_red[i][b] == 0 and pair_red[a][c] == 0:
                            if pair_red[i][a] and pair_red[i][c] and pair_red[b][a] and pair_red[b][c]:
                                old_viol += 1
                        if pair_red[i][c] == 0 and pair_red[a][b] == 0:
                            if pair_red[i][a] and pair_red[i][b] and pair_red[c][a] and pair_red[c][b]:
                                old_viol += 1
                    else:
                        # Blue: intra-pair IS red, cross NOT red
                        if pair_red[i][a] == 1 and pair_red[b][c] == 1:
                            if pair_red[i][b] == 0 and pair_red[i][c] == 0 and pair_red[a][b] == 0 and pair_red[a][c] == 0:
                                old_viol += 1
                        if pair_red[i][b] == 1 and pair_red[a][c] == 1:
                            if pair_red[i][a] == 0 and pair_red[i][c] == 0 and pair_red[b][a] == 0 and pair_red[b][c] == 0:
                                old_viol += 1
                        if pair_red[i][c] == 1 and pair_red[a][b] == 1:
                            if pair_red[i][a] == 0 and pair_red[i][b] == 0 and pair_red[c][a] == 0 and pair_red[c][b] == 0:
                                old_viol += 1
            
            # New color
            new_color = 1 - old_color
            if new_color == 1:
                other_same_new = [j for j in range(n_base) if j != i and current_x[j] == 1]
            else:
                other_same_new = [j for j in range(n_base) if j != i and current_x[j] == 0]
            
            new_viol = 0
            if len(other_same_new) >= 3:
                for tri in combinations(other_same_new, 3):
                    a, b, c = tri
                    if new_color == 1:
                        if pair_red[i][a] == 0 and pair_red[b][c] == 0:
                            if pair_red[i][b] and pair_red[i][c] and pair_red[a][b] and pair_red[a][c]:
                                new_viol += 1
                        if pair_red[i][b] == 0 and pair_red[a][c] == 0:
                            if pair_red[i][a] and pair_red[i][c] and pair_red[b][a] and pair_red[b][c]:
                                new_viol += 1
                        if pair_red[i][c] == 0 and pair_red[a][b] == 0:
                            if pair_red[i][a] and pair_red[i][b] and pair_red[c][a] and pair_red[c][b]:
                                new_viol += 1
                    else:
                        if pair_red[i][a] == 1 and pair_red[b][c] == 1:
                            if pair_red[i][b] == 0 and pair_red[i][c] == 0 and pair_red[a][b] == 0 and pair_red[a][c] == 0:
                                new_viol += 1
                        if pair_red[i][b] == 1 and pair_red[a][c] == 1:
                            if pair_red[i][a] == 0 and pair_red[i][c] == 0 and pair_red[b][a] == 0 and pair_red[b][c] == 0:
                                new_viol += 1
                        if pair_red[i][c] == 1 and pair_red[a][b] == 1:
                            if pair_red[i][a] == 0 and pair_red[i][b] == 0 and pair_red[c][a] == 0 and pair_red[c][b] == 0:
                                new_viol += 1
            
            delta_v = new_viol - old_viol
            new_v = current_v + delta_v
            
            if new_v < current_v:
                current_x[i] = new_color
                current_v = new_v
                if current_v == 0:
                    return current_x[:]
                if current_v < best_violations:
                    best_violations = current_v
                    best_x = current_x[:]
            else:
                # Simulated annealing
                if delta_v > 0:
                    import math
                    accept_prob = math.exp(-delta_v / max(temperature, 0.01))
                    if random.random() < accept_prob:
                        current_x[i] = new_color
                        current_v = new_v
                        if current_v < best_violations:
                            best_violations = current_v
                            best_x = current_x[:]
            
            temperature *= cooling_rate
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Build and Validate n=29 base
    base_adj = _build_paley29()
    
    # 2. Try to extend to n=30
    search_deadline = start_time + 115
    
    result_edges = _try_extend_to_30(base_adj, search_deadline)
    
    if result_edges is not None:
        # Build the 30-vertex graph
        n = 30
        adj = [0] * n
        for i in range(29):
            adj[i] = base_adj[i]
        for i in range(29):
            if result_edges[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        # Final verification before returning
        if _is_valid_coloring(adj, n):
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(1 if (adj[i] >> j) & 1 else 0)
            return {'n': n, 'colors': colors}
            
    # 3. Fallback: Return n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(1 if (base_adj[i] >> j) & 1 else 0)
            
    return {'n': n, 'colors': colors}
