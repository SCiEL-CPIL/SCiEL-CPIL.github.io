import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Build PG(2,5) polarity graph as base
    # Points in PG(2,5): triples (x,y,z) over GF(5), not all zero, modulo scaling
    # Represent each point by its canonical form (first nonzero entry is 1)
    
    def gf5_add(a, b): return (a + b) % 5
    def gf5_mul(a, b): return (a * b) % 5
    
    # Generate all points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: divide by first nonzero
                if x != 0:
                    canon = (1, gf5_mul(y, pow(x, 3, 5)), gf5_mul(z, pow(x, 3, 5)))
                elif y != 0:
                    canon = (0, 1, gf5_mul(z, pow(y, 3, 5)))
                else:
                    canon = (0, 0, 1)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    n_base = len(points)  # Should be 31
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity: point (x,y,z) is absolute if xz + y^2 = 0
    # Two points P=(x1,y1,z1) and Q=(x2,y2,z2) are adjacent if
    # x1*z2 + x2*z1 + y1*y2 = 0 (bilinear form)
    
    def are_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (gf5_mul(x1, z2) + gf5_mul(x2, z1) + gf5_mul(y1, y2)) % 5 == 0
    
    edges = []
    for i in range(n_base):
        for j in range(i+1, n_base):
            if are_adjacent(points[i], points[j]):
                edges.append([i, j])
    
    # Verify C4-free and get alpha
    def is_c4_free(n, edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def greedy_alpha(n, edge_list):
        """Upper bound on independence number via greedy coloring of complement"""
        # Actually, let's compute a lower bound on alpha via greedy
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Greedy: repeatedly pick vertex with min degree in remaining graph
        remaining = set(range(n))
        alpha = 0
        while remaining:
            # Find vertex with min degree in induced subgraph on remaining
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            alpha += 1
            remaining.remove(min_v)
            remaining -= adj[min_v]
        return alpha
    
    def exact_alpha_small(n, edge_list, timeout=5):
        """Exact alpha for small graphs using branch and bound"""
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        best = 0
        t0 = time.time()
        
        def branch(candidates, current):
            nonlocal best
            if time.time() - t0 > timeout:
                return
            if not candidates:
                if current > best:
                    best = current
                return
            if current + len(candidates) <= best:
                return
            # Pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            # Include v
            new_cand = candidates - {v} - adj[v]
            branch(new_cand, current + 1)
            # Exclude v
            branch(candidates - {v}, current)
        
        branch(set(range(n)), 0)
        return best
    
    # Check base graph
    if not is_c4_free(n_base, edges):
        # Fallback: use disjoint triangles
        fallback_edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            fallback_edges += [[a,b],[a,c],[b,c]]
        return {'n': 30, 'edges': fallback_edges}
    
    alpha_base = exact_alpha_small(n_base, edges, timeout=10)
    best_n = n_base
    best_edges = edges[:]
    
    if alpha_base > 10:
        # Need to reduce alpha - this shouldn't happen for polarity graph
        pass
    
    # Try to add vertices to the graph
    # A new vertex v can be added with a set of neighbors S such that:
    # 1. For any two existing vertices u1, u2: |N(u1) ∩ N(u2) ∩ S| ≤ 0 
    #    (i.e., no two existing vertices share a neighbor in S, combined with existing common neighbors)
    # 2. For any existing vertex u: |N(u) ∩ S| ≤ 1 (since u already has some common neighbors)
    #    Actually: for new vertex v with neighbors S, and existing vertex u:
    #    N(u) ∩ N(v) = N(u) ∩ S must have size ≤ 1
    # 3. For any two new vertices... we add one at a time so this is handled
    
    def can_add_vertex(n, edges, neighbor_set):
        """Check if we can add a new vertex with given neighbors"""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each existing vertex u, |N(u) ∩ neighbor_set| ≤ 1
        for u in range(n):
            common = adj[u] & neighbor_set
            if len(common) > 1:
                return False
        
        # For each pair of existing vertices u1, u2:
        # |N(u1) ∩ N(u2) ∩ neighbor_set| must be 0
        # (because N(u1) ∩ N(u2) already has ≤ 1 element, and adding more from neighbor_set
        #  would make it ≥ 2 if any element of neighbor_set is in both N(u1) and N(u2))
        # Actually: N(u1) ∩ N(u2) in the new graph = (N(u1) ∩ N(u2)) ∪ (N(u1) ∩ neighbor_set ∩ ...) 
        # Wait, the new vertex is only in N(u) for u in neighbor_set.
        # So for existing u1, u2: their common neighbors in the new graph =
        #   (old common neighbors) ∪ ({new_vertex} if u1 ∈ neighbor_set AND u2 ∈ neighbor_set)
        # So we need: if u1 ∈ neighbor_set AND u2 ∈ neighbor_set, then old |N(u1) ∩ N(u2)| ≤ 0
        # i.e., u1 and u2 must not already be adjacent to any common vertex
        # But they could be adjacent to each other - that's fine
        # The constraint is: for u1, u2 both in neighbor_set, |N(u1) ∩ N(u2)| must be 0
        
        for u1 in neighbor_set:
            for u2 in neighbor_set:
                if u1 == u2:
                    continue
                common = adj[u1] & adj[u2]
                if len(common) >= 1:
                    return False
        
        return True
    
    # Greedy: try to add vertices one by one
    n_current = n_base
    current_edges = edges[:]
    
    for attempt in range(12):  # Try to add up to 12 more vertices
        if time.time() > deadline - 30:
            break
        
        n_new = n_current + 1
        found = False
        
        # Try random neighbor sets
        for trial in range(2000):
            if time.time() > deadline - 30:
                break
            # Choose a random subset of existing vertices as neighbors
            # Size should be around n_current / (something)
            # For C4-free, degree should be O(sqrt(n))
            target_deg = max(2, int(n_new ** 0.5))
            k = random.randint(max(1, target_deg - 2), min(n_current, target_deg + 3))
            neighbors = set(random.sample(range(n_current), k))
            
            if can_add_vertex(n_current, current_edges, neighbors):
                # Add the vertex
                for nb in neighbors:
                    current_edges.append([n_current, nb])
                n_current = n_new
                found = True
                break
        
        if not found:
            break
    
    # Verify the result
    if is_c4_free(n_current, current_edges):
        alpha = exact_alpha_small(n_current, current_edges, timeout=15)
        if alpha <= 10:
            best_n = n_current
            best_edges = current_edges[:]
    
    # Local search: try to improve by swapping edges
    if time.time() < deadline - 20 and best_n < 39:
        # Try adding more vertices with local search
        n_ls = best_n
        ls_edges = best_edges[:]
        
        for outer in range(50):
            if time.time() > deadline - 10:
                break
            
            n_try = n_ls + 1
            improved = False
            
            for trial in range(500):
                if time.time() > deadline - 10:
                    break
                target_deg = max(2, int(n_try ** 0.5))
                k = random.randint(max(1, target_deg - 3), min(n_ls, target_deg + 4))
                neighbors = set(random.sample(range(n_ls), k))
                
                if can_add_vertex(n_ls, ls_edges, neighbors):
                    for nb in neighbors:
                        ls_edges.append([n_ls, nb])
                    n_ls = n_try
                    improved = True
                    break
            
            if not improved:
                break
        
        if is_c4_free(n_ls, ls_edges):
            alpha = exact_alpha_small(n_ls, ls_edges, timeout=15)
            if alpha <= 10 and n_ls > best_n:
                best_n = n_ls
                best_edges = ls_edges[:]
    
    # Final verification
    if best_n >= 1 and is_c4_free(best_n, best_edges):
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    fallback_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        fallback_edges += [[a,b],[a,c],[b,c]]
    return {'n': 30, 'edges': fallback_edges}
