import random
import time
import math

def construct():
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute all simple paths of length 3 (4 vertices, 3 edges)
    # Path is (s, u, v, t) with edges (s,u), (u,v), (v,t)
    # s, u, v, t are distinct
    # Note: The path is undirected, but the sequence s-u-v-t defines the structure.
    # The constraint is deg(s) == deg(t).
    # We generate unique sets of 3 edges forming a path.
    # To avoid duplicates, we can just iterate s, u, v, t and ensure the set of edges is unique?
    # Or just iterate all tuples and check?
    # Number of tuples is 15*14*13*12 = 32760.
    # Each path corresponds to 2 orientations? No, s-u-v-t and t-v-u-s are the same path structure.
    # But the condition deg(s)==deg(t) is symmetric.
    # So we can just iterate all permutations and check.
    
    paths = []
    for s in range(n):
        for u in range(n):
            if u == s: continue
            for v in range(n):
                if v == s or v == u: continue
                for t in range(n):
                    if t == s or t == u or t == v: continue
                    # Edges: (s,u), (u,v), (v,t)
                    # Normalize edges to (min, max)
                    e1 = (min(s, u), max(s, u))
                    e2 = (min(u, v), max(u, v))
                    e3 = (min(v, t), max(v, t))
                    # Ensure edges are distinct? 
                    # s!=u, u!=v, v!=t implies e1, e2, e3 are distinct?
                    # e1==e2 implies {s,u}=={u,v} => s=v (contradiction) or s=u (contradiction).
                    # So edges are distinct.
                    paths.append((e1, e2, e3, s, t))
    
    # There are duplicates in `paths` because (s,u,v,t) and (t,v,u,s) generate the same set of edges and same s,t pair (swapped).
    # Also (s,u,v,t) and (s',u',v',t') might generate same edges if the path is traversed differently?
    # No, a path of 3 edges is uniquely defined by its 4 vertices.
    # We can deduplicate by using a set of frozensets of edges?
    # But we need s and t for the degree check.
    # Actually, if we have edges {e1, e2, e3}, the endpoints of the path are the two vertices with degree 1 in the subgraph formed by these 3 edges.
    # So we can deduplicate based on the set of 3 edges.
    
    unique_paths = {}
    for e1, e2, e3, s, t in paths:
        key = frozenset([e1, e2, e3])
        if key not in unique_paths:
            unique_paths[key] = (s, t)
    
    # unique_paths maps frozenset(edges) -> (s, t)
    # Wait, is it possible that the same 3 edges form a path with different endpoints?
    # No, a tree of 3 edges has exactly 2 leaves.
    
    # Convert to list for iteration
    path_list = [(edges_set, s, t) for edges_set, (s, t) in unique_paths.items()]
    num_paths = len(path_list)
    
    def count_violations(edges, adj, deg):
        v_count = 0
        for edges_set, s, t in path_list:
            # Check if all 3 edges are in the graph
            # edges_set is a frozenset of 3 tuples
            # We can check adj quickly
            # Get the 3 edges
            e1, e2, e3 = tuple(edges_set)
            if adj[e1[0]][e1[1]] and adj[e2[0]][e2[1]] and adj[e3[0]][e3[1]]:
                if deg[s] == deg[t]:
                    v_count += 1
        return v_count

    def solve_sa():
        # Initialize with random sparse graph
        edges = set()
        adj = [[False]*n for _ in range(n)]
        deg = [0]*n
        
        # Add some random edges to start
        num_init = random.randint(20, 40)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        for e in shuffled_edges[:num_init]:
            u, v = e
            edges.add(e)
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            
        current_violations = count_violations(edges, adj, deg)
        current_edges = len(edges)
        
        best_edges = set()
        best_violations = current_violations
        best_edge_count = 0
        
        if current_violations == 0:
            best_edges = set(edges)
            best_edge_count = current_edges
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Time limit per run
        start_time = time.time()
        time_limit = 10.0
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Generate move
            # 50% Add, 50% Remove? Or Swap?
            # Let's try Swap: Remove one, Add one.
            # And pure Add/Remove.
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Pure Add
                if len(edges) < len(all_edges):
                    # Pick random non-edge
                    non_edges = [e for e in all_edges if e not in edges]
                    if non_edges:
                        e_add = random.choice(non_edges)
                        e_rem = None
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.6:
                # Pure Remove
                if edges:
                    e_rem = random.choice(list(edges))
                    e_add = None
                else:
                    continue
            else:
                # Swap
                if edges:
                    e_rem = random.choice(list(edges))
                    non_edges = [e for e in all_edges if e not in edges]
                    if non_edges:
                        e_add = random.choice(non_edges)
                    else:
                        continue
                else:
                    continue
            
            # Apply move
            new_edges = set(edges)
            new_adj = [row[:] for row in adj]
            new_deg = deg[:]
            
            if e_rem:
                u, v = e_rem
                new_edges.remove(e_rem)
                new_adj[u][v] = False
                new_adj[v][u] = False
                new_deg[u] -= 1
                new_deg[v] -= 1
            
            if e_add:
                u, v = e_add
                new_edges.add(e_add)
                new_adj[u][v] = True
                new_adj[v][u] = True
                new_deg[u] += 1
                new_deg[v] += 1
            
            new_violations = count_violations(new_edges, new_adj, new_deg)
            new_edge_count = len(new_edges)
            
            # Acceptance criteria
            # We want to minimize violations, then maximize edges.
            # If new_violations < current_violations: Accept
            # If new_violations == current_violations:
            #    If new_edge_count > current_edges: Accept
            #    If new_edge_count == current_edges: Accept (random walk)
            #    If new_edge_count < current_edges: Reject (or accept with prob exp(-1/T)?)
            # If new_violations > current_violations:
            #    Accept with prob exp(-(new_violations - current_violations) / T)
            
            accepted = False
            if new_violations < current_violations:
                accepted = True
            elif new_violations == current_violations:
                if new_edge_count >= current_edges:
                    accepted = True
                else:
                    # Allow moving to fewer edges if it helps escape? 
                    # Usually we want to stay at max edges.
                    # Let's reject to avoid drift.
                    accepted = False
            else:
                # new_violations > current_violations
                delta_v = new_violations - current_violations
                # Penalize increase in violations heavily
                prob = math.exp(-delta_v / T)
                if random.random() < prob:
                    accepted = True
            
            if accepted:
                edges = new_edges
                adj = new_adj
                deg = new_deg
                current_violations = new_violations
                current_edges = new_edge_count
                
                if current_violations == 0 and current_edges > best_edge_count:
                    best_edge_count = current_edges
                    best_edges = set(edges)
            
            T *= cooling_rate
            
        return list(best_edges)

    best_overall = []
    start_time = time.time()
    while time.time() - start_time < 50:
        res = solve_sa()
        if len(res) > len(best_overall):
            best_overall = res
    
    final_edges = []
    for u, v in best_overall:
        if isinstance(u, bool) or isinstance(v, bool):
            continue
        if not isinstance(u, int) or not isinstance(v, int):
            continue
        final_edges.append([int(u), int(v)])
        
    return final_edges
