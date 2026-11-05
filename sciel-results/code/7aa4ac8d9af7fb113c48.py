import time
import random
import sys
import numpy as np

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    Vertices are points of the plane.
    Two points p and r are adjacent if p lies on the polar line of r.
    This graph is C4-free.
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(adj, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses branch and bound with bitmasks for speed.
    """
    if n <= 64:
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        all_mask = (1 << n) - 1
        
        def _branch(rem_mask, current_size):
            if current_size >= limit:
                return True
            if rem_mask == 0:
                return False
            if current_size + rem_mask.bit_count() < limit:
                return False
            
            # Greedy pivot: pick vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            m = rem_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                deg = (adj_mask[v] & rem_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Include best_v
            rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
            if _branch(rem_include, current_size + 1):
                return True
            
            # Exclude best_v
            rem_exclude = rem_mask & ~(1 << best_v)
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(all_mask, 0)
    else:
        # Fallback for larger n (not expected in this task as n<=43)
        remaining = set(range(n))
        
        def _branch(rem, current_size):
            if current_size >= limit:
                return True
            if not rem:
                return False
            if current_size + len(rem) < limit:
                return False
            
            best_v = -1
            best_deg = -1
            for v in rem:
                deg = len(adj[v] & rem)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            rem_include = rem - adj[best_v] - {best_v}
            if _branch(rem_include, current_size + 1):
                return True
            
            rem_exclude = rem - {best_v}
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(remaining, 0)

def _find_valid_neighbor_set(adj, n, new_vertex_idx, target_size=None):
    """
    Find a set of neighbors S for a new vertex such that
    for all u, v in S, N(u) intersect N(v) is empty.
    We want to maximize the size of S or just find one that works for alpha.
    Here we just generate a few random valid sets.
    """
    candidates = list(range(n))
    random.shuffle(candidates)
    
    S = set()
    for v in candidates:
        valid = True
        for u in S:
            if adj[u] & adj[v]:
                valid = False
                break
        if valid:
            S.add(v)
            
    return S

def _find_valid_neighbor_set_systematic(adj, n):
    """
    Systematically find a valid neighbor set for a new vertex.
    A valid set S must satisfy: for all u, v in S, N(u) ∩ N(v) = ∅.
    This is equivalent to S being an independent set in the "conflict graph"
    where two vertices conflict if they share a common neighbor.
    
    We use a greedy approach with multiple random orderings to find large valid sets.
    """
    # Build conflict graph: u and v conflict if they share a common neighbor
    # For efficiency, we'll check on-the-fly
    
    best_S = set()
    
    # Try multiple random orderings
    for trial in range(50):
        candidates = list(range(n))
        random.shuffle(candidates)
        
        S = set()
        for v in candidates:
            valid = True
            for u in S:
                if adj[u] & adj[v]:
                    valid = False
                    break
            if valid:
                S.add(v)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    return best_S

def _try_add_vertex(adj, n, S):
    """
    Try to add a new vertex connected to all vertices in S.
    Returns (new_adj, new_n) if successful, else (None, None).
    """
    if not S:
        return None, None
    
    new_adj = [set(s) for s in adj]
    new_adj.append(set(S))
    for u in S:
        new_adj[u].add(n)
    
    # Verify C4-freeness (should be guaranteed by construction if S is valid)
    # But let's double-check for safety
    for u in S:
        for v in S:
            if u < v:
                common = new_adj[u] & new_adj[v]
                if len(common) >= 2:
                    return None, None
    
    return new_adj, n + 1

def _local_search_optimize(adj, n, deadline, max_iterations=1000):
    """
    Try to improve the graph by local search:
    - Remove edges to reduce independence number
    - Add edges to maintain C4-freeness while reducing alpha
    """
    best_adj = [set(s) for s in adj]
    best_n = n
    improved = True
    
    while improved and time.time() < deadline - 10:
        improved = False
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Try adding a random edge if it maintains C4-freeness and reduces alpha
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or v in adj[u]:
                continue
            
            # Check if adding edge (u, v) maintains C4-freeness
            common_neighbors = adj[u] & adj[v]
            if len(common_neighbors) >= 1:
                # Would create a C4, skip
                continue
            
            # Tentatively add edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Check if this reduces independence number (or at least doesn't increase it)
            if not _max_independent_set(adj, n, limit=11):
                # Good, keep this edge
                improved = True
            else:
                # Revert
                adj[u].remove(v)
                adj[v].remove(u)
    
    return adj, n

def construct():
    deadline = time.time() + 140.0
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    if not _verify_c4_free(n, adj):
        return {'n': 0, 'edges': []}
        
    if _max_independent_set(adj, n, limit=11):
        return {'n': 0, 'edges': []}
        
    best_n = n
    best_edges = edges[:]
    
    # Systematically try to add vertices
    max_vertices_to_add = 12
    for _ in range(max_vertices_to_add):
        if time.time() > deadline - 30:
            break
            
        added = False
        # Try multiple times with different random seeds
        for attempt in range(200):
            if time.time() > deadline - 15:
                break
                
            # Find a valid neighbor set
            S = _find_valid_neighbor_set_systematic(adj, n)
            
            if not S:
                continue
            
            # Try to add vertex with this neighbor set
            new_adj, new_n = _try_add_vertex(adj, n, S)
            if new_adj is None:
                continue
            
            # Check if the new graph has no independent set of size 11
            if not _max_independent_set(new_adj, new_n, limit=11):
                adj = new_adj
                n = new_n
                for u in S:
                    edges.append([n - 1, u])
                
                if n > best_n:
                    best_n = n
                    best_edges = [e[:] for e in edges]
                    
                added = True
                break
        
        if not added:
            break
    
    # Local search to try to add more edges (reducing alpha further)
    if time.time() < deadline - 20:
        adj, n = _local_search_optimize(adj, n, deadline)
        
        # Reconstruct edges from adj
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        
        if n > best_n:
            best_n = n
            best_edges = edges[:]
    
    return {'n': best_n, 'edges': best_edges}
