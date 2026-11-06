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
