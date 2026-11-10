"""Combined program: Fast bitmask C6 check + Greedy + 2-Swap Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # Safety margin before 150s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """
    Check if adding edge (u,v) creates a C6.
    Path u - x1 - x2 - x3 - x4 - v.
    x1 in N(v) \ {u}
    x4 in N(u) \ {v, x1}
    x2 in N(x1) \ {v, x4, u}
    x3 in N(x4) \ {u, x2, v, x1}
    Edge (x2, x3) must exist.
    """
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            
            # Early exit: if N(x1) \ {u,v,x4} and N(x4) \ {u,v,x1} have no common vertices, skip
            x2_candidates = adj[x1] & ~(1 << v) & ~(1 << x4) & ~(1 << u)
            x3_candidates = adj[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
            
            # Check if there's any x2 in x2_candidates that has an edge to some x3 in x3_candidates
            # i.e., check if (adj[x2] & x3_candidates) != 0 for any x2 in x2_candidates
            while x2_candidates:
                x2 = (x2_candidates & (-x2_candidates)).bit_length() - 1
                x2_candidates &= x2_candidates - 1
                
                # x3 in N(x4) \ {u, x2, v, x1}
                x3_mask = x3_candidates & ~(1 << x2)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    
                    if adj[x2] & (1 << x3):
                        return False  # C6 found
    return True  # No C6 through this edge

def _is_c6_free(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_through_edge(adj_bits, i, j, n):
                    return False
    return True

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=2.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _spectral_gap_score(adj, u, v, n=40):
    """
    Degree-based heuristic: prefer edges between lower-degree vertices
    to balance the graph and avoid creating dense clusters that lead to C6s.
    """
    deg_u = bin(adj[u]).count('1')
    deg_v = bin(adj[v]).count('1')
    
    # Common neighbors
    common = adj[u] & adj[v]
    common_count = bin(common).count('1')
    
    # Prefer lower degree sum (balance the graph)
    score = -(deg_u + deg_v) * 0.1
    
    # Penalize common neighbors (triangles are bad for C6-free)
    score -= common_count * 0.5
    
    # Bonus for connecting very different degree vertices (expansion)
    if abs(deg_u - deg_v) >= 2:
        score += 0.3
    
    return score

def _local_search_2swap(adj, n=40, max_time=10.0):
    """
    Local search using 1-add and 2-swap moves with degree-based prioritization.
    """
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    
    stuck_counter = 0
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try to add edges (1-add) with degree-based prioritization
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # Score non-edges: prefer lower degree sum
        if len(non_edges) > 0:
            scored = [(e, bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1')) for e in non_edges]
            scored.sort(key=lambda x: x[1])  # Ascending by degree sum
            non_edges = [e for e, _ in scored]
        
        # Try in order, but with some randomization for diversity
        candidates = non_edges[:]
        random.shuffle(candidates[5:])  # Keep top 5 in order, shuffle rest
        
        for u, v in candidates:
            if time.time() - start_time > max_time:
                return best_adj, best_count
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _check_c6_through_edge(adj, u, v, n):
                current += 1
                improved = True
                if current > best_count:
                    best_count = current
                    best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if improved:
            stuck_counter = 0
            no_improve_count = 0
            continue
            
        # No edges could be added
        no_improve_count += 1
        stuck_counter += 1
        
        if stuck_counter >= 8:
            break
        
        # 2. Perturbation: remove edges from high-degree vertices to enable new additions
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
        
        # Target edges incident to high-degree vertices for removal
        deg = [bin(adj[i]).count('1') for i in range(n)]
        edge_deg_sum = [(e, deg[e[0]] + deg[e[1]]) for e in cur_edges]
        edge_deg_sum.sort(key=lambda x: -x[1])  # Descending by degree sum
        high_deg_edges = [e for e, _ in edge_deg_sum[:min(10, len(edge_deg_sum))]]
        
        num_remove = 2 if stuck_counter < 4 else random.choice([2, 3])
        
        perturbations_tried = 0
        max_perturbations = 300
        
        while perturbations_tried < max_perturbations and time.time() - start_time < max_time:
            perturbations_tried += 1
            
            if len(high_deg_edges) < num_remove:
                break
            remove_edges = random.sample(high_deg_edges, num_remove)
            
            if len(non_edges) < num_remove:
                break
            add_edges = random.sample(non_edges, num_remove)
            
            saved_state = adj[:]
            for e in remove_edges:
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
            
            valid = True
            for e in add_edges:
                adj[e[0]] |= (1 << e[1])
                adj[e[1]] |= (1 << e[0])
                if not _check_c6_through_edge(adj, e[0], e[1], n):
                    valid = False
                    break
            
            if valid:
                improved = True
                cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                break
            else:
                adj = saved_state[:]
        
        if not improved and stuck_counter >= 6:
            break

    return best_adj, best_count

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search with 2-swaps from 8xK5
    if time.time() - start_time < DEADLINE - 10:
        adj_ls, count_ls = _local_search_2swap(adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 5))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts + 2-swap
    for attempt in range(25):
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 2:
            break
        
        # Short greedy
        mt_greedy = min(1.0, remaining / 5)
        adj_g, count_g = _random_greedy(n, seed=500 + attempt, max_time=mt_greedy)
        
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
            
        # Apply local search on the greedy result
        if count_g >= best_edges - 6:
            mt_ls = min(4.0, remaining / 3)
            adj_l, count_l = _local_search_2swap(adj_g, n, max_time=mt_ls)
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
