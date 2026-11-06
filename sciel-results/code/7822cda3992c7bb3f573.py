import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    n = 40
    best_edges = []
    best_count = 0
    
    def check_c6_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) is a path of length 5 from u to v in G.
        We check if v is reachable from u in exactly 5 steps without using (u,v).
        Since (u,v) is not yet in the graph, we just check path length 5.
        """
        # BFS for exactly 5 steps from u, see if we reach v
        # Using bitmasks for speed
        visited = 0
        current = 1 << u
        visited |= current
        
        for step in range(5):
            next_frontier = 0
            c = current
            while c:
                lsb = c & -c
                node = lsb.bit_length() - 1
                next_frontier |= adj[node]
                c ^= lsb
            
            # Remove already visited to ensure simple path
            next_frontier &= ~visited
            
            if next_frontier & (1 << v):
                return True
            
            visited |= next_frontier
            current = next_frontier
            if current == 0:
                return False
        
        return False
    
    def greedy_fill(adj, edges_list, edge_set, deadline):
        """Try to add all possible missing edges that don't create C6."""
        added = 0
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    return added
                if not (adj[u] & (1 << v)):
                    if not check_c6_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
                        added += 1
        return added
    
    def local_search(adj, edges_list, edge_set, deadline, max_iters=5000):
        """Local search: try edge swaps to increase edge count."""
        current_count = len(edges_list)
        best_local = current_count
        best_adj = [a for a in adj]
        best_edges = list(edges_list)
        best_set = set(edge_set)
        
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            # Randomly pick an edge to remove
            if not edges_list:
                break
            idx = random.randint(0, len(edges_list) - 1)
            u_rem, v_rem = edges_list[idx]
            
            # Save state
            save_adj = [a for a in adj]
            save_edges = list(edges_list)
            save_set = set(edge_set)
            
            # Remove the edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            edges_list.remove([u_rem, v_rem])
            edge_set.discard((u_rem, v_rem))
            
            # Try to add as many missing edges as possible
            added_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        break
                    if not (adj[u] & (1 << v)):
                        if not check_c6_fast(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges_list.append([u, v])
                            edge_set.add((u, v))
                            added_edges.append([u, v])
                
                if time.time() > deadline:
                    break
            
            new_count = len(edges_list)
            
            if new_count > best_local:
                best_local = new_count
                best_adj = [a for a in adj]
                best_edges = list(edges_list)
                best_set = set(edge_set)
            elif new_count <= current_count:
                # Revert
                adj = save_adj
                edges_list = save_edges
                edge_set = save_set
            else:
                # Keep (neutral or slight improvement)
                current_count = new_count
        
        return best_adj, best_edges, best_set, best_local
    
    def try_initialization(seed_type, deadline):
        """Try different initializations and return best result."""
        adj = [0] * n
        edges_list = []
        edge_set = set()
        
        if seed_type == 0:
            # 8 disjoint K5s
            for k in range(8):
                base = 5 * k
                for i in range(5):
                    for j in range(i + 1, 5):
                        u = base + i
                        v = base + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
        elif seed_type == 1:
            # Random graph with p=0.4
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.4:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
        elif seed_type == 2:
            # Bipartite-like: connect based on modular arithmetic
            # Split into two groups of 20
            for u in range(20):
                for v in range(20, 40):
                    if (u * 7 + v * 3) % 5 != 0:  # Sparse bipartite
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
        elif seed_type == 3:
            # Complete graph on 20 vertices + random edges to other 20
            for u in range(20):
                for v in range(u + 1, 20):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges_list.append([u, v])
                    edge_set.add((u, v))
            for u in range(20):
                for v in range(20, 40):
                    if random.random() < 0.3:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
        elif seed_type == 4:
            # Start empty, pure greedy
            pass
        elif seed_type == 5:
            # Two K10s + random cross edges
            for group in [range(0, 20), range(20, 40)]:
                for i, u in enumerate(group):
                    for j, v in enumerate(group):
                        if i < j:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges_list.append([u, v])
                            edge_set.add((u, v))
            for u in range(20):
                for v in range(20, 40):
                    if random.random() < 0.25:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_list.append([u, v])
                        edge_set.add((u, v))
        
        # Greedy fill
        greedy_fill(adj, edges_list, edge_set, deadline)
        
        # Local search
        adj, edges_list, edge_set, count = local_search(adj, edges_list, edge_set, deadline, max_iters=2000)
        
        return edges_list, count
    
    # Try multiple initializations with different seeds
    random.seed(42)
    
    for seed_type in [0, 4, 1, 5, 2, 3]:
        if time.time() > deadline - 5:
            break
        
        edges_list, count = try_initialization(seed_type, deadline)
        
        if count > best_count:
            best_count = count
            best_edges = list(edges_list)
    
    # Final local search on best result if time permits
    if time.time() < deadline - 10 and best_edges:
        adj = [0] * n
        edge_set = set()
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
        
        adj, edges_list, edge_set, count = local_search(adj, best_edges, edge_set, deadline, max_iters=3000)
        if count > best_count:
            best_count = count
            best_edges = list(edges_list)
    
    return best_edges
