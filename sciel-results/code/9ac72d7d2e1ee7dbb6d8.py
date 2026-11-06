import time
import random
import numpy as np

def count_c4(adj, n):
    """Count C4s: for each pair of adjacent vertices, count pairs of common neighbors."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if i < j:
                common = len(adj[i] & adj[j])
                count += common * (common - 1) // 2
    return count

def greedy_remove_c4s(adj, n, deadline, max_iters=5000):
    """Greedily remove edges to eliminate all C4s. Returns (adj, edges_list, c4_count)."""
    # Build edge list
    edges = set()
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.add((i, j))
    
    # Find all C4s and their edges
    c4_edges = {}  # edge -> list of C4s containing it (represented by the 4 vertices)
    
    # For efficiency, find C4s by checking pairs of common neighbors
    for i in range(n):
        if time.time() > deadline:
            break
        for j in list(adj[i]):
            if i >= j:
                continue
            common = adj[i] & adj[j]
            common_list = list(common)
            for a_idx in range(len(common_list)):
                a = common_list[a_idx]
                for b_idx in range(a_idx + 1, len(common_list)):
                    b = common_list[b_idx]
                    # C4: i-a-j-b-i
                    c4 = frozenset([i, a, j, b])
                    # The 4 edges of this C4
                    c4_edges.setdefault((i, a), []).append(c4)
                    c4_edges.setdefault((a, j), []).append(c4)
                    c4_edges.setdefault((j, b), []).append(c4)
                    c4_edges.setdefault((b, i), []).append(c4)
    
    # Greedily remove edges that appear in the most C4s
    removed = 0
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find edge in most C4s
        best_edge = None
        best_count = 0
        for edge, c4s in c4_edges.items():
            if len(c4s) > best_count:
                best_count = len(c4s)
                best_edge = edge
        
        if best_edge is None or best_count == 0:
            break
        
        u, v = best_edge
        # Remove this edge
        adj[u].remove(v)
        adj[v].remove(u)
        edges.discard((u, v))
        removed += 1
        
        # Update c4_edges: remove this edge and all C4s containing it
        for c4 in c4_edges.get((u, v), []):
            for e in [(i, j) for i, j in [(min(c), max(c)) for c in c4] if False]:
                pass
            # Remove this C4 from all other edges in the C4
            c4_list = sorted(c4)
            # The 4 edges of the C4: (c4_list[0],c4_list[1]), (c4_list[1],c4_list[2]), etc.
            # Actually, a C4 is a cycle, so we need the actual cycle edges
            # Let's just remove the C4 from all edges that had it
            for edge_key in list(c4_edges.keys()):
                if c4 in c4_edges[edge_key]:
                    c4_edges[edge_key].remove(c4)
                    if not c4_edges[edge_key]:
                        del c4_edges[edge_key]
        
        if (u, v) in c4_edges:
            del c4_edges[(u, v)]
        if (v, u) in c4_edges:
            del c4_edges[(v, u)]
    
    return adj, list(edges), removed

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    n = 31
    best_c4 = float('inf')
    best_edges = None
    
    # Strategy: Try multiple random initializations, greedily remove C4s,
    # then add the 91st edge optimally, then local search.
    
    random.seed(42)
    
    num_trials = 0
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        num_trials += 1
        
        # Random dense graph: each edge with probability ~0.8 to start with ~120+ edges
        p = 0.75 + 0.05 * (trial % 5)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Greedily remove C4s until C4-free
        adj, edges, removed = greedy_remove_c4s(adj, n, deadline - 15)
        
        if len(edges) < 90:
            continue
        
        # If we have more than 90 edges, we need to get to exactly 90 C4-free
        # Since greedy removal should have made it C4-free, if >90, remove random edges
        while len(edges) > 90:
            # Remove a random edge
            e = random.choice(edges)
            u, v = e
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove(e)
        
        if len(edges) != 90:
            continue
        
        # Verify C4-free (should be, but check)
        c4_count = count_c4(adj, n)
        if c4_count > 0:
            # Try to fix by removing more edges
            for _ in range(20):
                if time.time() > deadline - 15:
                    break
                # Find a C4
                found = False
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            common = adj[i] & adj[j]
                            if len(common) >= 2:
                                common_list = list(common)
                                a, b = common_list[0], common_list[1]
                                # Remove edge (i,j)
                                adj[i].remove(j)
                                adj[j].remove(i)
                                edges.remove((i, j))
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
            if len(edges) < 90:
                continue
            c4_count = count_c4(adj, n)
            if c4_count > 0:
                continue
        
        # Now we have a 90-edge C4-free graph. Add the 91st edge.
        # Find the non-edge that creates the fewest C4s
        min_delta = float('inf')
        best_add = None
        for i in range(n):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, n):
                if j in adj[i]:
                    continue
                common = len(adj[i] & adj[j])
                delta = common * (common - 1) // 2
                if delta < min_delta:
                    min_delta = delta
                    best_add = (i, j)
        
        if best_add is None:
            continue
        
        u, v = best_add
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
        current_c4 = count_c4(adj, n)
        
        # Local search: edge swaps
        for ls_iter in range(30):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for rem_idx in range(len(edges)):
                if time.time() > deadline - 10:
                    break
                
                ru, rv = edges[rem_idx]
                common_ru = adj[ru] & adj[rv]
                destroyed = len(common_ru) * (len(common_ru) - 1) // 2
                
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                
                best_net = 0
                best_add2 = None
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i + 1, n):
                        if j in adj[i]:
                            continue
                        if i == ru and j == rv:
                            continue
                        common_ij = len(adj[i] & adj[j])
                        created = common_ij * (common_ij - 1) // 2
                        net = created - destroyed
                        if net < best_net:
                            best_net = net
                            best_add2 = (i, j)
                
                adj[ru].add(rv)
                adj[rv].add(ru)
                
                if best_add2 is not None and best_net < 0:
                    a, b = best_add2
                    adj[ru].remove(rv)
                    adj[rv].remove(ru)
                    adj[a].add(b)
                    adj[b].add(a)
                    edges[rem_idx] = (a, b)
                    current_c4 += best_net
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges = [[e[0], e[1]] for e in edges]
                    improved = True
            
            if not improved:
                break
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = [[e[0], e[1]] for e in edges]
    
    if best_edges is None:
        # Fallback: polarity graph
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, 4, 5)
                        pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                    elif y != 0:
                        inv_y = pow(y, 4, 5)
                        pts = (0, 1, (z * inv_y) % 5)
                    else:
                        pts = (0, 0, 1)
                    if pts not in points:
                        points.append(pts)
        
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                u = points[i]
                v = points[j]
                b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
                if b_val == 0:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        # Add best edge
        min_delta = float('inf')
        best_add = None
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    continue
                common = len(adj[i] & adj[j])
                delta = common * (common - 1) // 2
                if delta < min_delta:
                    min_delta = delta
                    best_add = [i, j]
        
        edges.append(best_add)
        best_edges = edges
        best_c4 = count_c4(adj, n)
    
    return best_edges
