<<<<<<< SEARCH
import random
import time
import copy

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, 5s safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, c, u, v):
        # Check if adding (u, v) is valid
        # 1. c[u][v] <= 2
        if c[u][v] > 2:
            return False
        
        # 2. For all w in N(u), c[w][v] <= 1
        for w in adj[u]:
            if c[w][v] > 1:
                return False
        
        # 3. For all w in N(v), c[u][w] <= 1
        for w in adj[v]:
            if c[u][w] > 1:
                return False
        
        return True

    def add_edge(state, u, v):
        """Add edge (u, v) to state in-place. State = (adj, c, edges)"""
        adj, c, edges = state
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        c[u][v] += 1
        c[v][u] += 1
        for w in adj[u]:
            c[w][v] += 1
            c[v][w] += 1
        for w in adj[v]:
            c[u][w] += 1
            c[w][u] += 1

    def remove_edge(state, u, v):
        """Remove edge (u, v) from state in-place. State = (adj, c, edges)"""
        adj, c, edges = state
        # Remove from edges list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break
        adj[u].remove(v)
        adj[v].remove(u)
        c[u][v] -= 1
        c[v][u] -= 1
        for w in adj[u]:
            c[w][v] -= 1
            c[v][w] -= 1
        for w in adj[v]:
            c[u][w] -= 1
            c[w][u] -= 1

    def build_state(edges):
        adj = [set() for _ in range(n)]
        c = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            c[u][v] += 1
            c[v][u] += 1
            for w in adj[u]:
                c[w][v] += 1
                c[v][w] += 1
            for w in adj[v]:
                c[u][w] += 1
                c[w][u] += 1
        return adj, c, list(edges)

    def greedy_construct():
        """Run greedy construction from scratch."""
        current_edges = []
        current_adj = [set() for _ in range(n)]
        current_c = [[0] * n for _ in range(n)]
        state = (current_adj, current_c, current_edges)
        
        missing_edges = list(all_edges)
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, current_c, u, v):
                add_edge(state, u, v)
        
        return current_edges

    def local_search(edges):
        """Local search with edge swaps to escape local optima."""
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        current_adj, current_c, current_edges = build_state(edges)
        state = (current_adj, current_c, current_edges)
        
        # Phase 1: Try to add edges greedily
        for _ in range(50):
            if time.time() > deadline:
                break
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            added_any = False
            for u, v in missing:
                if is_valid_addition(current_adj, current_c, u, v):
                    add_edge(state, u, v)
                    added_any = True
            if not added_any:
                break
        
        if len(current_edges) > best_local_count:
            best_local_count = len(current_edges)
            best_local_edges = list(current_edges)
        
        # Phase 2: Edge swaps (remove one, try to add multiple)
        max_swaps = 200
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            if not current_edges:
                break
            
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            # Save state
            saved_edges = list(current_edges)
            saved_adj = [set(s) for s in current_adj]
            saved_c = [row[:] for row in current_c]
            
            # Remove edge
            remove_edge(state, u, v)
            
            # Try to add edges
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            for a, b in missing:
                if is_valid_addition(current_adj, current_c, a, b):
                    add_edge(state, a, b)
            
            if len(current_edges) > best_local_count:
                best_local_count = len(current_edges)
                best_local_edges = list(current_edges)
            else:
                # Revert
                current_edges = saved_edges
                current_adj = saved_adj
                current_c = saved_c
                state = (current_adj, current_c, current_edges)
        
        return best_local_edges

    def simulated_annealing(edges):
        """Simulated annealing with edge swaps."""
        current_adj, current_c, current_edges = build_state(edges)
        state = (current_adj, current_c, current_edges)
        
        best_local_edges = list(current_edges)
        best_local_count = len(current_edges)
        
        T = 1.0
        cooling = 0.995
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Try to add an edge
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    if is_valid_addition(current_adj, current_c, u, v):
                        add_edge(state, u, v)
                        if len(current_edges) > best_local_count:
                            best_local_count = len(current_edges)
                            best_local_edges = list(current_edges)
                        break
            
            # Try a swap: remove one edge, add another
            if current_edges and random.random() < 0.5:
                idx = random.randint(0, len(current_edges) - 1)
                u, v = current_edges[idx]
                
                saved_edges = list(current_edges)
                saved_adj = [set(s) for s in current_adj]
                saved_c = [row[:] for row in current_c]
                
                remove_edge(state, u, v)
                
                missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
                random.shuffle(missing)
                for a, b in missing:
                    if is_valid_addition(current_adj, current_c, a, b):
                        add_edge(state, a, b)
                        break
                
                if len(current_edges) >= best_local_count:
                    if len(current_edges) > best_local_count:
                        best_local_count = len(current_edges)
                        best_local_edges = list(current_edges)
                else:
                    # Revert
                    current_edges = saved_edges
                    current_adj = saved_adj
                    current_c = saved_c
                    state = (current_adj, current_c, current_edges)
            
            T *= cooling
        
        return best_local_edges

    # Main loop
    iterations = 0
    last_local_search = 0
    last_sa = 0
    
    while time.time() < deadline:
        # Run a greedy construction
        edges = greedy_construct()
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        
        iterations += 1
        
        # Every 200 iterations, run local search on the best found so far
        if iterations % 200 == 0 and time.time() - last_local_search > 3.0:
            last_local_search = time.time()
            improved = local_search(best_edges)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved
        
        # Every 500 iterations, run simulated annealing
        if iterations % 500 == 0 and time.time() - last_sa > 5.0:
            last_sa = time.time()
            improved = simulated_annealing(best_edges)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved

    return best_edges
=======
import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, 5s safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Bitmask-based adjacency for O(1) K2,3 checks
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) maintains K2,3-free property using bitmasks."""
        # After adding (u,v), common neighbors of any pair (x,y) must be <= 2
        # Key insight: adding (u,v) affects common-neighbor counts for pairs
        # involving u or v, and for pairs (w1,w2) where w1 in N(u), w2 in N(v)
        
        # Common neighbors of u and v after addition = |N(u) ∩ N(v)|
        # This must be <= 2
        common_uv = adj_mask[u] & adj_mask[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # For each w in N(u): common neighbors of w and v after addition
        # = |N(w) ∩ N(v) ∪ {u}| = |N(w) ∩ N(v)| + 1 (since u is now in N(v))
        # But u might already be in N(w)... no, w in N(u) means u in N(w)
        # Actually: N(v) gains u. So common(w,v) = |N(w) ∩ (N(v) ∪ {u})|
        # = |N(w) ∩ N(v)| + [1 if u in N(w) and u not in N(v)]
        # Since w in N(u), u in N(w). So:
        # common(w,v) = |N(w) ∩ N(v)| + 1 if u not in N(v), else |N(w) ∩ N(v)|
        # We need this <= 2 for all w in N(u)
        
        for w in range(n):
            if adj_mask[u] >> w & 1:  # w in N(u)
                # common(w, v) after adding (u,v)
                if adj_mask[v] >> u & 1:
                    # u already in N(v) (shouldn't happen if we're adding it)
                    common_wv = bin(adj_mask[w] & adj_mask[v]).count('1')
                else:
                    common_wv = bin(adj_mask[w] & adj_mask[v]).count('1') + 1
                if common_wv > 2:
                    return False
        
        # Similarly for w in N(v): common(u, w) after addition
        for w in range(n):
            if adj_mask[v] >> w & 1:  # w in N(v)
                if adj_mask[u] >> v & 1:
                    common_uw = bin(adj_mask[u] & adj_mask[w]).count('1')
                else:
                    common_uw = bin(adj_mask[u] & adj_mask[w]).count('1') + 1
                if common_uw > 2:
                    return False
        
        return True

    def add_edge_fast(adj_mask, edge_set, u, v):
        """Add edge (u,v) to bitmask adjacency and edge set."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def remove_edge_fast(adj_mask, edge_set, u, v):
        """Remove edge (u,v) from bitmask adjacency and edge set."""
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edge_set.discard((u, v))
        edge_set.discard((v, u))
    
    def edges_from_mask(adj_mask):
        """Convert bitmask adjacency to list of edges."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] >> v & 1:
                    edges.append([u, v])
        return edges
    
    def structured_initial():
        """Create a structured initial graph based on known constructions."""
        adj_mask = [0] * n
        edge_set = set()
        
        # Start with a 5-clique structure repeated
        # Use a known dense K2,3-free construction approach
        # Try: complete graph on 5 vertices, then connect carefully
        
        # Strategy: Use the incidence structure of a projective plane or
        # a carefully designed regular graph
        
        # Let's try a 4-regular or 5-regular construction
        # For K2,3-free, we need any two vertices to share at most 2 neighbors
        
        # Try: partition into 4 groups of 5, make each group a clique,
        # then add inter-group edges carefully
        groups = [[i for i in range(g*5, (g+1)*5)] for g in range(4)]
        
        # Make each group a clique
        for g in range(4):
            for i in range(5):
                for j in range(i+1, 5):
                    u = groups[g][i]
                    v = groups[g][j]
                    if can_add_edge(adj_mask, u, v):
                        add_edge_fast(adj_mask, edge_set, u, v)
        
        # Add inter-group edges: connect corresponding vertices across groups
        # For vertices in different groups, try to add edges
        for g1 in range(4):
            for g2 in range(g1+1, 4):
                for i in range(5):
                    u = groups[g1][i]
                    v = groups[g2][i]
                    if can_add_edge(adj_mask, u, v):
                        add_edge_fast(adj_mask, edge_set, u, v)
        
        # Try to add more edges
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    if can_add_edge(adj_mask, u, v):
                        add_edge_fast(adj_mask, edge_set, u, v)
        
        return edges_from_mask(adj_mask), adj_mask, edge_set
    
    def greedy_construct():
        """Run greedy construction with bitmask acceleration."""
        adj_mask = [0] * n
        edge_set = set()
        
        missing_edges = list(all_edges)
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if can_add_edge(adj_mask, u, v):
                add_edge_fast(adj_mask, edge_set, u, v)
        
        return edges_from_mask(adj_mask), adj_mask, edge_set
    
    def local_search(adj_mask, edge_set):
        """Local search with edge swaps using bitmask acceleration."""
        current_edges = edges_from_mask(adj_mask)
        best_edges = list(current_edges)
        best_count = len(current_edges)
        
        # Phase 1: Try to add edges greedily
        for _ in range(100):
            if time.time() > deadline:
                break
            added_any = False
            missing = [e for e in all_edges if e[0] not in edge_set.__class__ and (e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set]
            random.shuffle(missing)
            for u, v in missing:
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    if can_add_edge(adj_mask, u, v):
                        add_edge_fast(adj_mask, edge_set, u, v)
                        added_any = True
            if not added_any:
                break
        
        current_edges = edges_from_mask(adj_mask)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
        # Phase 2: Edge swaps (remove one, try to add multiple)
        max_swaps = 300
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            current_edges = edges_from_mask(adj_mask)
            if not current_edges:
                break
            
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            # Save state
            saved_adj_mask = list(adj_mask)
            saved_edge_set = set(edge_set)
            
            # Remove edge
            remove_edge_fast(adj_mask, edge_set, u, v)
            
            # Try to add edges
            added_count = 0
            missing = [e for e in all_edges if (e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set]
            random.shuffle(missing)
            for a, b in missing:
                if (a, b) not in edge_set and (b, a) not in edge_set:
                    if can_add_edge(adj_mask, a, b):
                        add_edge_fast(adj_mask, edge_set, a, b)
                        added_count += 1
            
            current_edges = edges_from_mask(adj_mask)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
            else:
                # Revert
                adj_mask = saved_adj_mask
                edge_set = saved_edge_set
        
        return best_edges, adj_mask, edge_set
    
    def simulated_annealing(adj_mask, edge_set):
        """Simulated annealing with edge swaps using bitmask acceleration."""
        current_edges = edges_from_mask(adj_mask)
        best_edges = list(current_edges)
        best_count = len(current_edges)
        
        T = 1.0
        cooling = 0.995
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Try to add an edge
            missing = [e for e in all_edges if (e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        if can_add_edge(adj_mask, u, v):
                            add_edge_fast(adj_mask, edge_set, u, v)
                            current_edges = edges_from_mask(adj_mask)
                            if len(current_edges) > best_count:
                                best_count = len(current_edges)
                                best_edges = list(current_edges)
                            break
            
            # Try a swap: remove one edge, add another
            if random.random() < 0.6:
                current_edges = edges_from_mask(adj_mask)
                if current_edges:
                    idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[idx]
                    
                    saved_adj_mask = list(adj_mask)
                    saved_edge_set = set(edge_set)
                    
                    remove_edge_fast(adj_mask, edge_set, u, v)
                    
                    missing = [e for e in all_edges if (e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set]
                    random.shuffle(missing)
                    for a, b in missing:
                        if (a, b) not in edge_set and (b, a) not in edge_set:
                            if can_add_edge(adj_mask, a, b):
                                add_edge_fast(adj_mask, edge_set, a, b)
                                break
                    
                    current_edges = edges_from_mask(adj_mask)
                    if len(current_edges) >= best_count:
                        if len(current_edges) > best_count:
                            best_count = len(current_edges)
                            best_edges = list(current_edges)
                    else:
                        # Revert
                        adj_mask = saved_adj_mask
                        edge_set = saved_edge_set
            
            T *= cooling
        
        return best_edges, adj_mask, edge_set
    
    # Main loop
    # Start with structured initial
    best_edges, best_adj_mask, best_edge_set = structured_initial()
    best_count = len(best_edges)
    
    iterations = 0
    last_local_search = 0
    last_sa = 0
    
    while time.time() < deadline:
        # Run a greedy construction
        edges, adj_mask, edge_set = greedy_construct()
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj_mask = adj_mask
            best_edge_set = edge_set
        
        iterations += 1
        
        # Every 100 iterations, run local search on the best found so far
        if iterations % 100 == 0 and time.time() - last_local_search > 2.0:
            last_local_search = time.time()
            improved, adj_mask, edge_set = local_search(best_adj_mask, best_edge_set)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved
                best_adj_mask = adj_mask
                best_edge_set = edge_set
        
        # Every 200 iterations, run simulated annealing
        if iterations % 200 == 0 and time.time() - last_sa > 4.0:
            last_sa = time.time()
            improved, adj_mask, edge_set = simulated_annealing(best_adj_mask, best_edge_set)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved
                best_adj_mask = adj_mask
                best_edge_set = edge_set

    return best_edges
>>>>>>> REPLACE
