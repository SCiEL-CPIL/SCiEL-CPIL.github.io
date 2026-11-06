import random
import time
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Build PG(2,5) polarity graph
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0
    
    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
    
    # Find best edge to add to reach 91
    def get_c4s_for_edge(u, v, adj_sets):
        """Find all C4s containing edge (u,v)."""
        c4s = set()
        # Type 1: (u,v) is a side: u-v-x-y-u
        for x in adj_sets[v]:
            if x == u:
                continue
            for y in adj_sets[u]:
                if y == v or y == x:
                    continue
                if x in adj_sets[y]:
                    c4 = frozenset([u, v, x, y])
                    c4s.add(c4)
        # Type 2: (u,v) is a diagonal: u-x-v-y-u
        common = adj_sets[u] & adj_sets[v]
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                x = common_list[i]
                y = common_list[j]
                c4 = frozenset([u, v, x, y])
                c4s.add(c4)
        return c4s
    
    # Add the best edge
    best_add = None
    min_c4s_created = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                c4s = get_c4s_for_edge(i, j, adj)
                if len(c4s) < min_c4s_created:
                    min_c4s_created = len(c4s)
                    best_add = (i, j)
    
    u_add, v_add = best_add
    edges.add((u_add, v_add))
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    
    # Initialize all C4s
    all_c4s = set()
    for u, v in edges:
        all_c4s.update(get_c4s_for_edge(u, v, adj))
    
    current_c4_count = len(all_c4s)
    best_c4_count = current_c4_count
    best_edges = set(edges)
    best_adj = [set(s) for s in adj]
    best_c4s = set(all_c4s)
    
    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper: check if pair is an edge (handles both orders)
    def is_edge(u, v, edge_set):
        return (u, v) in edge_set or (v, u) in edge_set
    
    # Local search with exact C4 tracking
    iterations = 0
    while time.time() < deadline and iterations < 100000:
        iterations += 1
        
        if not edges:
            break
        
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Find C4s containing (u1, v1)
        c4s_to_remove = get_c4s_for_edge(u1, v1, adj)
        
        improved = False
        # Try multiple random additions
        for _ in range(50):
            u2, v2 = random.choice(all_pairs)
            if is_edge(u2, v2, edges):
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            # Compute C4s that would be added by (u2, v2)
            # We need to consider the graph AFTER removing (u1, v1)
            # Modify adj temporarily for computation
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            
            c4s_to_add = get_c4s_for_edge(u2, v2, adj)
            
            # Restore adj for now
            adj[u1].add(v1)
            adj[v1].add(u1)
            
            # Exact delta computation
            removed_only = c4s_to_remove - c4s_to_add
            added_only = c4s_to_add - c4s_to_remove
            
            new_c4_count = current_c4_count - len(removed_only) + len(added_only)
            
            if new_c4_count < best_c4_count:
                # Accept the swap
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.discard((u1, v1))
                edges.add((u2, v2))
                
                # Update C4 set exactly
                all_c4s.difference_update(c4s_to_remove)
                all_c4s.update(c4s_to_add)
                
                current_c4_count = len(all_c4s)
                improved = True
                
                if current_c4_count < best_c4_count:
                    best_c4_count = current_c4_count
                    best_edges = set(edges)
                    best_adj = [set(s) for s in adj]
                    best_c4s = set(all_c4s)
                
                break
        
        # If no improvement in this pass, try a random restart with perturbation
        if not improved and time.time() < deadline - 20:
            # Perturb: remove k random edges, add k random non-edges
            k = random.randint(3, 6)
            edge_list = list(edges)
            random.shuffle(edge_list)
            for _ in range(min(k, len(edge_list))):
                e = edge_list.pop()
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                edges.discard(e)
            
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if not is_edge(i, j, edges)]
            random.shuffle(non_edges)
            for _ in range(min(k, len(non_edges))):
                e = non_edges.pop()
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
                edges.add(e)
            
            # Recompute all C4s from scratch
            all_c4s = set()
            for u, v in edges:
                all_c4s.update(get_c4s_for_edge(u, v, adj))
            current_c4_count = len(all_c4s)
            
            if current_c4_count < best_c4_count:
                best_c4_count = current_c4_count
                best_edges = set(edges)
                best_adj = [set(s) for s in adj]
                best_c4s = set(all_c4s)
    
    # Return best result
    best_edges_list = [[u, v] for u, v in sorted(best_edges)]
    return best_edges_list
