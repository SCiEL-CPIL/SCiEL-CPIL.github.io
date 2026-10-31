<<<<<<< SEARCH
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    # Target N close to Moore bound 512
    # Try N = 520, 522, ...
    # k=3
    
    N = 520
    
    # 1. Generate random 3-regular graph
    edges = _generate_random_3reg(N)
    if edges is None:
        # Fallback to lift
        return _construct_lift(k, g)
        
    # 2. Repair loop
    adj = _build_adj(N, edges)
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Check girth
        min_cycle, cycle_nodes = _find_short_cycle(adj, g)
        
        if min_cycle >= g:
            return N, edges
            
        # Repair: Pick an edge in the short cycle to break
        # We need to find a 2-switch that increases girth or doesn't decrease it too much
        # Simple heuristic: Random 2-switch involving one of the cycle edges
        
        # Pick a random edge from the cycle
        # The cycle is a list of nodes [u0, u1, ..., uk]
        # Edges are (u0,u1), (u1,u2), ...
        
        # Let's pick a random edge (u, v) from the cycle
        idx = random.randint(0, len(cycle_nodes) - 1)
        u = cycle_nodes[idx]
        v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
        
        # Find another edge (w, x) not in the cycle and not sharing nodes with u,v
        # to perform a switch.
        # Actually, we can just pick any other edge in the graph.
        
        # To be efficient, we can pick a random edge from the whole graph
        # and check if it's valid for a switch.
        
        # A 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x)
        # Or (u,x) and (v,w).
        
        # We need to ensure (u,w) and (v,x) are not existing edges.
        
        # Let's try a few random switches
        success = False
        for _ in range(100):
            # Pick random edge (w, x)
            # We need a list of all edges
            # To make this fast, we need to maintain a set of edges
            
            # This is getting complicated to maintain state.
            # Let's restart the graph generation if repair fails?
            # No, let's just try to improve the current one.
            
            pass # Placeholder

    # If repair fails, try Lift
    return _construct_lift(k, g)

def _generate_random_3reg(N):
    # Pairing model
    # 3N half-edges
    # Randomly pair them
    # Check for loops and multi-edges
    # Retry if invalid
    
    for _ in range(1000):
        half_edges = []
        for i in range(N):
            for _ in range(3):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = set()
        valid = True
        for i in range(0, 3*N, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if valid:
            return list(edges)
            
    return None

def _build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_short_cycle(adj, g):
    N = len(adj)
    # BFS from each node
    # Return length and nodes of shortest cycle found < g
    
    # To speed up, we can stop BFS at depth g//2
    # If we find a back edge, we have a cycle.
    
    best_len = g
    best_cycle = []
    
    for s in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            if dist[u] >= g//2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Cycle found
                    # Length = dist[u] + dist[v] + 1
                    len_c = dist[u] + dist[v] + 1
                    if len_c < best_len:
                        best_len = len_c
                        # Reconstruct cycle
                        # This is expensive, so we only do it if it's the best so far
                        # Or we can just return the length and find the cycle later?
                        # We need the nodes to repair.
                        # Let's reconstruct
                        cycle = []
                        # Path from u to LCA? No, in BFS tree, parent pointers lead to root.
                        # The cycle is formed by the tree paths from u and v to their LCA, plus the edge (u,v).
                        # Actually, in a BFS tree, if v is already visited and not parent,
                        # the cycle is u-v-...-LCA-...-u.
                        
                        # Simple reconstruction:
                        # Trace back from u and v to find common ancestor?
                        # Or just trace back from u to root and v to root?
                        
                        # For repair, we just need ONE edge from the cycle.
                        # We can just return the edge (u,v) as the "bad" edge?
                        # No, (u,v) is a tree edge? No, (u,v) is the non-tree edge that closes the cycle.
                        # So (u,v) is definitely in the cycle.
                        # We can just use (u,v) for repair.
                        
                        return len_c, [u, v] # Just return the edge endpoints for now?
                        # Wait, the function returns cycle_nodes.
                        # Let's just return the edge (u,v) as the cycle "hint".
                        # The repair logic can just use (u,v).
                        
    return best_len, best_cycle
=======
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Moore bound for n(3, 16) is 512.
    # We will try to construct a graph with N vertices where N is even (required for 3-regular).
    # Start close to the bound.
    
    # Try a few values of N
    # 512 is the Moore bound. Is it achievable?
    # For k=3, g=16, the cage order is likely larger.
    # Let's try N = 512, 514, 516...
    
    # However, random 3-regular graphs with N=512 will have many short cycles.
    # We need a strategy that produces high girth.
    
    # Strategy:
    # 1. Generate random 3-regular graph.
    # 2. Use local search (2-switches) to increase girth.
    # 3. If stuck, try a different N or restart.
    
    # To be safe and fast, we might just return a known valid graph if we had one,
    # but we don't. We must generate.
    
    # Let's try N = 512 first.
    N = 512
    
    # 1. Generate random 3-regular graph
    edges = _generate_random_3reg(N)
    if edges is None:
        # Fallback: try larger N
        N = 514
        edges = _generate_random_3reg(N)
        if edges is None:
            # Absolute fallback: return a trivially valid but large graph?
            # No, must be simple 3-regular.
            # If we can't generate, we fail. But random gen rarely fails for N>10.
            raise ValueError("Failed to generate random 3-regular graph")

    adj = _build_adj(N, edges)
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # 2. Repair loop: 2-switches to increase girth
    # We want to eliminate cycles shorter than g.
    
    max_iterations = 2000
    current_girth = _compute_girth(adj, g)
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Check if current girth is sufficient
        if current_girth >= g:
            return N, edges
            
        # Find a short cycle
        cycle = _find_short_cycle(adj, g)
        if not cycle:
            # Should not happen if current_girth < g
            break
            
        # Cycle is a list of nodes [v0, v1, ..., vk]
        # Perform a 2-switch on an edge in the cycle.
        # Pick a random edge (u, v) from the cycle.
        m = len(cycle)
        idx = random.randint(0, m - 1)
        u = cycle[idx]
        v = cycle[(idx + 1) % m]
        
        # Find another edge (w, x) to switch with.
        # To avoid creating new short cycles or multi-edges, we should be careful.
        # A simple heuristic: pick a random edge (w, x) from the graph.
        # Ensure (w, x) is not in the cycle (or at least not adjacent to u, v in a bad way).
        # Actually, just picking a random edge and checking validity is okay.
        
        # To be more effective, we might want to pick an edge "far" from the cycle.
        # But for now, random is fine.
        
        # Pick random edge (w, x)
        # We need a list of edges to pick from.
        # We can maintain a list of edges.
        # Let's just pick random vertices w, x and check if (w,x) is an edge.
        # This is inefficient. Better to maintain edge list.
        
        # Let's maintain edge_list and edge_set.
        # We have edge_set. We can convert to list for random choice, but that's O(E).
        # E = 3N/2 = 768. That's small.
        # So we can just pick a random edge from the list.
        
        # Let's create a list of edges from the set once, or maintain it.
        # For simplicity, let's just pick random vertices and check if they are connected.
        # Probability of hitting an edge is ~ 3N/N^2 = 3/N ~ 0.006.
        # Too low.
        
        # Let's maintain a list of edges.
        # We'll do it inside the loop if it's fast enough.
        # Or just use the edges list we have.
        
        # We have `edges` list. But it changes.
        # Let's maintain `edge_list` and `edge_set` in sync.
        
        # Initialize edge_list
        if not hasattr(construct, '_edge_list_cache'):
            pass # Just a placeholder, we will manage it locally
        
        # Actually, let's just re-extract edge list from set every iteration? No, O(E).
        # E is small. 768 edges.
        # Let's just do:
        edge_list = list(edge_set)
        
        # Pick random edge (w, x) from edge_list
        # Ensure it's not (u, v)
        # And ensure w, x are not u, v (to avoid triangles or multi-edges immediately)
        
        valid_switch = False
        for _ in range(50): # Try a few random edges
            w, x = random.choice(edge_list)
            if w == u or w == v or x == u or x == v:
                continue
            if (w, x) == (u, v) or (w, x) == (v, u):
                continue
            
            # Check if (u, w) and (v, x) are valid new edges
            # They must not exist
            # And they must not be loops (u!=w, v!=x which is true since w,x not u,v)
            
            if (u, w) in edge_set or (v, x) in edge_set:
                continue
            if (u, x) in edge_set or (v, w) in edge_set:
                continue
                
            # Perform the switch: remove (u,v) and (w,x), add (u,w) and (v,x)
            # Note: We must also consider the other pairing: (u,x) and (v,w).
            # Let's try (u,w) and (v,x) first.
            
            # Remove old edges
            edge_set.remove((min(u,v), max(u,v)))
            edge_set.remove((min(w,x), max(w,x)))
            
            # Add new edges
            new_e1 = (min(u,w), max(u,w))
            new_e2 = (min(v,x), max(v,x))
            
            # Check if new edges create multi-edges?
            # We already checked they are not in edge_set.
            # But what if new_e1 == new_e2?
            # That would mean (u,w) == (v,x) => u==v and w==x or u==x and w==v.
            # u!=v, w!=x.
            # If u==x and w==v, then (u,v) and (w,x) are the same edge, which we excluded.
            # So new_e1 != new_e2.
            
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            # Update adjacency list
            # This is tricky to do incrementally efficiently without a full rebuild.
            # Given N=512, rebuilding adj is O(N+E) ~ 1200 ops.
            # 2000 iterations * 1200 = 2.4M ops. Fast enough.
            
            adj = _build_adj(N, list(edge_set))
            
            # Check if this switch improved girth or kept it
            new_girth = _compute_girth(adj, g)
            
            if new_girth > current_girth:
                current_girth = new_girth
                valid_switch = True
                break
            elif new_girth == current_girth:
                # Accept with some probability to allow exploration?
                # Or just accept to move?
                # If we don't accept, we might get stuck.
                # Let's accept if it doesn't decrease.
                valid_switch = True
                break
            else:
                # Revert
                edge_set.remove(new_e1)
                edge_set.remove(new_e2)
                edge_set.add((min(u,v), max(u,v)))
                edge_set.add((min(w,x), max(w,x)))
                adj = _build_adj(N, list(edge_set))
                
        if not valid_switch:
            # No good switch found in 50 tries
            # Try a different edge from the cycle?
            # Or just restart the whole process?
            # Let's just continue the loop, picking a different edge from the cycle next time.
            pass

    # If we exit the loop without success, return the best graph we have
    return N, list(edge_set)

def _generate_random_3reg(N):
    # Pairing model
    for _ in range(1000):
        half_edges = []
        for i in range(N):
            for _ in range(3):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = set()
        valid = True
        for i in range(0, 3*N, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if valid:
            return list(edges)
            
    return None

def _build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _compute_girth(adj, max_girth):
    """
    Compute the girth of the graph, stopping if it exceeds max_girth.
    Returns the shortest cycle length found, or max_girth if none shorter found.
    """
    N = len(adj)
    min_cycle = max_girth
    
    for s in range(N):
        # BFS from s
        dist = [-1] * N
        parent = [-1] * N
        dist[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            if dist[u] >= min_cycle // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Cycle found
                    # Length = dist[u] + dist[v] + 1
                    len_c = dist[u] + dist[v] + 1
                    if len_c < min_cycle:
                        min_cycle = len_c
                        if min_cycle == 3: # Can't get smaller
                            return 3
                        # Update limit
                        # We can't easily update the BFS limit in the middle,
                        # but we can break early if we find a very short cycle.
    
    return min_cycle

def _find_short_cycle(adj, g):
    """
    Find a cycle of length < g.
    Returns list of nodes in the cycle, or None.
    """
    N = len(adj)
    
    for s in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Cycle found
                    len_c = dist[u] + dist[v] + 1
                    if len_c < g:
                        # Reconstruct cycle
                        # Path from u to LCA and v to LCA
                        # Simple method: trace back from u and v to root?
                        # No, LCA is not necessarily root.
                        
                        # Let's trace back from u and v to find the intersection.
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                                
                        if lca is None:
                            # Should not happen in a tree
                            continue
                            
                        # Construct cycle
                        # Path u -> lca
                        idx_u = path_u.index(lca)
                        path_u_to_lca = path_u[:idx_u+1]
                        
                        # Path v -> lca
                        idx_v = path_v.index(lca)
                        path_v_to_lca = path_v[:idx_v+1]
                        
                        # Cycle: u -> ... -> lca -> ... -> v -> u
                        # path_u_to_lca is [u, ..., lca]
                        # path_v_to_lca is [v, ..., lca]
                        # We need to reverse path_v_to_lca to get lca -> ... -> v
                        path_lca_to_v = list(reversed(path_v_to_lca))
                        
                        cycle = path_u_to_lca + path_lca_to_v[1:]
                        return cycle
                        
    return None
>>>>>>> REPLACE
