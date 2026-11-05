import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
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
    
    def count_new_c4s(u, v, adj):
        count = 0
        nu = adj[u]
        nv = adj[v]
        for a in nu:
            for b in nv:
                if a == b:
                    continue
                if b in adj[a]:
                    count += 1
        return count

    def count_total_c4(adj):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2

    # Ensure we have exactly 91 edges
    if m < 91:
        # Greedily add edges
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        while len(edges) < 91:
            best_cost = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        cost = count_new_c4s(i, j, adj)
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (i, j)
            
            if best_edge:
                u, v = best_edge
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
            else:
                break

    elif m > 91:
        # Greedily remove edges with highest C4 participation
        # This part is less likely to be triggered for standard polarity graph
        # But we handle it for robustness
        # Calculate participation
        # For simplicity, just remove random edges if stuck, or greedy
        # Let's try greedy removal based on local C4 count
        while len(edges) > 91:
            min_cost = float('inf')
            min_edge = None
            for e in edges:
                u, v = e[0], e[1]
                # Cost of removing is the number of C4s this edge is in
                # This is count_new_c4s(u,v) if we were adding, but for removal it's the same count
                # Actually, removing an edge removes all C4s containing it.
                # So we want to remove edges with HIGH C4 count.
                cost = count_new_c4s(u, v, adj)
                if cost > min_cost: # We want max cost to remove
                    # Wait, logic error in variable name. 
                    # Let's find max participation.
                    pass
            
            # Correct logic:
            max_part = -1
            max_edge = None
            for e in edges:
                u, v = e[0], e[1]
                part = count_new_c4s(u, v, adj)
                if part > max_part:
                    max_part = part
                    max_edge = e
            
            if max_edge:
                u, v = max_edge[0], max_edge[1]
                # Remove from edges list
                edges.remove(max_edge)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                break

    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    # Local Search / Simulated Annealing
    # Initialize current state
    current_adj = [set(s) for s in adj]
    current_score = count_total_c4(current_adj)
    
    best_adj = [set(s) for s in current_adj]
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute all possible non-edges for faster random selection?
    # Or just pick random u,v and check.
    
    while time.time() < deadline - 2.0:
        # Generate random move
        # Pick random edge to remove
        # To do this efficiently, we can pick a random vertex u, then a random neighbor v.
        u = random.randint(0, n-1)
        if not current_adj[u]:
            continue
        v = random.choice(list(current_adj[u]))
        
        # Pick random non-edge to add
        # Pick random x, y
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y:
            continue
        if y in current_adj[x]:
            continue
            
        # Ensure we don't remove the edge we are adding (if x==u and y==v)
        # Since (u,v) is an edge and (x,y) is a non-edge, they are distinct.
        
        # Apply move
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        current_adj[x].add(y)
        current_adj[y].add(x)
        
        new_score = count_total_c4(current_adj)
        
        # Acceptance
        if new_score < best_score:
            best_score = new_score
            best_adj = [set(s) for s in current_adj]
            
        if new_score <= current_score or random.random() < np.exp((current_score - new_score) / T):
            # Accept move (keep current_adj as is)
            current_score = new_score
        else:
            # Reject move, revert
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_adj[x].remove(y)
            current_adj[y].remove(x)
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck?
            # For now, just keep going.

    # Convert best_adj back to edges list
    best_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    return best_edges
