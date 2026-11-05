import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n = 31
    q = 5
    mod = q
    
    # Enumerate all points in PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    
    # The polarity graph for q=5 should have 90 edges.
    # If not, we might need to adjust, but the problem statement implies this is the core.
    # Let's verify the count.
    # If m != 90, we might have a bug in construction or the graph is not the one intended.
    # However, ex(31, C4)=90 is a known value.
    
    if m != 90:
        # Fallback: If construction failed, try to build a random 90-edge C4-free graph?
        # Or just proceed with what we have and add/remove to get 91.
        # But for now, assume m=90.
        pass

    # We need to add exactly 1 edge to reach 91.
    # We want to minimize the number of C4s created.
    # Cost of adding (u, v) is the number of C4s containing (u, v).
    # A C4 containing (u, v) is u-v-a-b-u.
    # This requires a in N(v), b in N(u), and (a, b) in E.
    # So cost = number of edges between N(u) and N(v).
    
    def count_c4s_created(u, v, adj):
        nu = adj[u]
        nv = adj[v]
        count = 0
        # For each a in N(v), count neighbors of a that are in N(u)
        for a in nv:
            # Neighbors of a
            for b in adj[a]:
                if b in nu:
                    count += 1
        return count

    # Find the non-edge with minimum cost
    min_cost = float('inf')
    best_edge = None
    edge_set = set()
    for e in edges:
        if e[0] < e[1]:
            edge_set.add((e[0], e[1]))
        else:
            edge_set.add((e[1], e[0]))

    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                cost = count_c4s_created(i, j, adj)
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (i, j)

    if best_edge:
        u, v = best_edge
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)

    # Now we have 91 edges.
    # We can try to improve by local search (swap edges) if time permits.
    # But the single addition to the extremal graph is likely optimal or very close.
    # Let's do a quick local search to be safe.
    
    def count_total_c4(adj):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2

    current_score = count_total_c4(adj)
    best_score = current_score
    best_adj = [set(s) for s in adj]
    
    # Simple local search: try swapping edges
    # Pick a random edge (u, v) to remove, and a random non-edge (x, y) to add.
    # Accept if score decreases.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # Precompute list of edges for random removal
    current_edges = list(edges)
    
    while time.time() < deadline - 2.0:
        # Remove random edge
        if not current_edges: break
        idx = random.randint(0, len(current_edges)-1)
        u, v = current_edges[idx]
        
        # Add random non-edge
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y: continue
        if y in adj[x]: continue
        # Ensure we are not just adding the edge we just removed (if x==u and y==v)
        # It's a swap, so it's fine.
        
        # Apply
        adj[u].remove(v)
        adj[v].remove(u)
        current_edges.pop(idx)
        
        adj[x].add(y)
        adj[y].add(x)
        current_edges.append([x, y])
        
        new_score = count_total_c4(adj)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = [set(s) for s in adj]
        
        if new_score <= current_score or random.random() < np.exp((current_score - new_score) / T):
            current_score = new_score
        else:
            # Revert
            adj[x].remove(y)
            adj[y].remove(x)
            current_edges.pop()
            
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append([u, v])
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Convert best_adj to edge list
    best_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    return best_edges
