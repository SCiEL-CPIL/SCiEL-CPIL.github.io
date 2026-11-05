import random
import time

def construct():
    # Baseline: n=39, empty graph
    best_n = 39
    best_edges = []
    
    # Try to find a graph for n=46
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Local search for n=46
    n = 46
    min_deg_required = n - 39  # 7
    
    # Initialize random graph
    # We want min degree 7, so average degree maybe 8-10
    p = 0.2 
    edges = set()
    # Random graph generation
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edges.add((u, v))
                
    # Function to check C4
    def has_c4(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check for C4
        # For each pair of nodes, check common neighbors
        # If two nodes have 2 common neighbors, they form a C4 with those?
        # No, if u and v have common neighbors w1, w2, then u-w1-v-w2-u is a C4.
        # So we just need to check if any pair has >= 2 common neighbors.
        for u in range(n):
            for v in range(u+1, n):
                common = set(adj[u]).intersection(adj[v])
                if len(common) >= 2:
                    return True, (u, v, list(common)[:2])
        return False, None

    # Function to get min degree
    def get_min_deg(edges, n):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg)

    # Iterative removal
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        c4_exists, c4_info = has_c4(edges, n)
        if not c4_exists:
            # Check min degree
            if get_min_deg(edges, n) >= min_deg_required:
                return {'n': n, 'edges': [list(e) for e in edges]}
            else:
                # Add edges to low degree nodes
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                low_nodes = [i for i in range(n) if deg[i] < min_deg_required]
                if low_nodes:
                    u = random.choice(low_nodes)
                    # Try to add edge to a random node
                    v = random.randint(0, n-1)
                    if u != v and (u, v) not in edges and (v, u) not in edges:
                        # Check if adding creates C4
                        new_edges = edges | {(u, v) if u < v else (v, u)}
                        c4_check, _ = has_c4(new_edges, n)
                        if not c4_check:
                            edges = new_edges
                            continue
                # If can't add, just continue removing or restart?
                # For now, just continue loop (might remove other C4s)
                if c4_exists:
                     # Remove an edge from C4
                     u, v, commons = c4_info
                     # Remove edge (u, v) or (u, c1) etc?
                     # Removing (u, v) might not break C4 if it's not part of it?
                     # The C4 is u-c1-v-c2-u. Edges are (u,c1), (c1,v), (v,c2), (c2,u).
                     # (u,v) is a diagonal, not in C4.
                     # So we must remove one of the cycle edges.
                     # Let's remove (u, c1).
                     edge_to_remove = (u, commons[0]) if u < commons[0] else (commons[0], u)
                     edges.discard(edge_to_remove)
                continue
        else:
            # Remove edge from C4
            u, v, commons = c4_info
            # C4 is u-c1-v-c2-u
            # Edges: (u, c1), (c1, v), (v, c2), (c2, u)
            # Pick one randomly
            c1 = commons[0]
            c2 = commons[1]
            candidates = [
                (u, c1), (c1, v), (v, c2), (c2, u)
            ]
            edge_to_remove = random.choice(candidates)
            edge_to_remove = (edge_to_remove[0], edge_to_remove[1]) if edge_to_remove[0] < edge_to_remove[1] else (edge_to_remove[1], edge_to_remove[0])
            edges.discard(edge_to_remove)

    # Fallback: return seed
    return {'n': best_n, 'edges': best_edges}
