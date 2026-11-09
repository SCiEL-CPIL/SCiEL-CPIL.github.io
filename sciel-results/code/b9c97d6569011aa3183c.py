"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    # Represent points as homogeneous coordinates [x:y:z] over GF(5)
    # Normalize: first non-zero coordinate is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)  # p[i]^(5-2) mod 5
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)  # should be 31
    assert n == 31, f"Expected 31 points, got {n}"
    
    # Polar form: for point p=[a,b,c], the polar line is ax + by + cz = 0
    # Two points p, q are adjacent if p lies on polar line of q (or equivalently q on polar of p)
    # In PG(2,5) with this polarity, p·q = 0 (dot product mod 5)
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _check_c4_free(adj, n):
    """Check if graph is C4-free: no two vertices share more than one common neighbor."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _max_independent_set_size(adj, n, time_limit=2.0):
    """
    Find the size of the maximum independent set using branch and bound.
    Returns the size.
    """
    start_time = time.time()
    best = [0]
    
    # Use a greedy coloring bound for the branch and bound
    # Independent set in G = clique in complement
    # We'll use a simple recursive approach with pruning
    
    # Order vertices by degree (heuristic)
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -deg[i])
    
    # For each vertex, list its non-neighbors (candidates for independent set)
    non_neighbors = [set(range(n)) - adj[i] - {i} for i in range(n)]
    
    def _branch(candidates, current_size, current_set):
        if time.time() - start_time > time_limit:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best[0]:
            return
        
        # Pick a vertex from candidates
        v = next(iter(candidates))
        
        # Include v
        new_candidates = candidates & non_neighbors[v]
        _branch(new_candidates, current_size + 1, current_set | {v})
        
        # Exclude v
        new_candidates2 = candidates - {v}
        _branch(new_candidates2, current_size, current_set)
    
    _branch(set(range(n)), 0, set())
    return best[0]

def _greedy_extend(edges, adj, n, target_alpha=10, time_limit=100.0):
    """
    Try to add vertices one by one, maintaining C4-freeness and alpha < target_alpha+1.
    """
    start_time = time.time()
    current_edges = list(edges)
    current_adj = [set(s) for s in adj]
    current_n = n
    
    # Try to add vertices
    while time.time() - start_time < time_limit:
        # Try adding a new vertex with various connection patterns
        added = False
        # Try connecting the new vertex to subsets of existing vertices
        # The new vertex v_new must satisfy: for any two existing vertices u, w both adjacent to v_new,
        # u and w must not already share a common neighbor (otherwise C4)
        # Actually: v_new adjacent to u and w creates C4 if u and w share a neighbor other than v_new.
        # So: for the set S of neighbors of v_new, no two vertices in S can share a common neighbor in the existing graph.
        
        # Greedy: try small sets of neighbors first
        for size in range(0, min(current_n, 8)):
            if time.time() - start_time > time_limit:
                break
            # Try random subsets of size 'size'
            for trial in range(200):
                if time.time() - start_time > time_limit:
                    break
                if size == 0:
                    neighbors = set()
                else:
                    neighbors = set(random.sample(range(current_n), size))
                
                # Check C4-freeness: for any two vertices u, w in neighbors, 
                # they must not share a common neighbor in current graph
                valid = True
                if len(neighbors) >= 2:
                    neighbor_list = list(neighbors)
                    for idx_u in range(len(neighbor_list)):
                        for idx_w in range(idx_u + 1, len(neighbor_list)):
                            u, w = neighbor_list[idx_u], neighbor_list[idx_w]
                            common = current_adj[u] & current_adj[w]
                            if len(common) > 0:
                                valid = False
                                break
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                # Check independence number: adding v_new with neighbors set
                # v_new is in an independent set iff it's not adjacent to any vertex in the set
                # So we need to check if alpha increases to 11
                # Quick check: if there exists an independent set of size 10 not containing v_new,
                # and v_new is not adjacent to any of them, then alpha becomes 11
                # For efficiency, just check: is there an independent set of size 10 in the 
                # subgraph induced by non-neighbors of v_new?
                
                # Simpler: just build the new graph and check alpha
                new_n = current_n + 1
                new_adj = [set(s) for s in current_adj]
                new_adj.append(set(neighbors))
                for u in neighbors:
                    new_adj[u].add(new_n - 1)
                
                # Quick C4 check on new graph
                c4_ok = True
                for i in range(new_n):
                    for j in range(i+1, new_n):
                        common = new_adj[i] & new_adj[j]
                        if len(common) > 1:
                            c4_ok = False
                            break
                    if not c4_ok:
                        break
                
                if not c4_ok:
                    continue
                
                # Check alpha
                alpha = _max_independent_set_size(new_adj, new_n, time_limit=1.5)
                if alpha >= 11:
                    continue
                
                # Valid! Add the vertex
                current_n = new_n
                current_adj = new_adj
                for u in neighbors:
                    current_edges.append([u, new_n - 1])
                added = True
                break
            if added:
                break
        
        if not added:
            break
    
    return current_n, current_edges, current_adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Build the polarity graph of PG(2,5)
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    # Verify C4-free
    assert _check_c4_free(adj, n), "Polarity graph should be C4-free"
    
    # Verify alpha
    alpha = _max_independent_set_size(adj, n, time_limit=5.0)
    
    # Try to extend
    if time.time() < deadline - 10:
        n2, edges2, adj2 = _greedy_extend(edges, adj, n, target_alpha=10, time_limit=deadline - time.time() - 15)
        if n2 > n:
            n, edges, adj = n2, edges2, adj2
            alpha = _max_independent_set_size(adj, n, time_limit=5.0)
    
    # Final validation
    assert _check_c4_free(adj, n), "Final graph must be C4-free"
    assert alpha < 11, f"Independence number must be < 11, got {alpha}"
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
