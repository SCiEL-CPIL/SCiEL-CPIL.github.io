import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    N = 40
    M = 401
    
    def get_initial_graph():
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        edges = all_edges[:M]
        
        adj = [0] * N
        edge_set = set()
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            
        return adj, edge_set

    def compute_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            common = adj[u] & adj[v]
            books[(u, v)] = bin(common).count('1')
        return books

    def max_book(books):
        if not books:
            return 0
        return max(books.values())

    def local_search(adj, edge_set, time_limit):
        books = compute_books(adj, edge_set)
        current_max = max_book(books)
        best_max_in_run = current_max
        best_state = (adj.copy(), set(edge_set), best_max_in_run)
        
        stuck_counter = 0
        
        while time.time() < time_limit:
            if current_max <= 7:
                break
            
            # Try to improve
            improved = False
            # Pick a random edge to remove, biased towards high book size
            # To avoid biasing too much, pick from top 20% or just random.
            # Let's pick an edge with book size == current_max.
            max_edges = [e for e, b in books.items() if b == current_max]
            if not max_edges:
                max_edges = list(books.keys())
            
            u, v = random.choice(max_edges)
            
            # Try a few random swaps
            for _ in range(10):
                x = random.randint(0, N - 1)
                y = random.randint(0, N - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                
                if (x, y) in edge_set:
                    continue
                if (x, y) == (u, v):
                    continue
                
                # Perform Swap
                # Remove (u, v)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edge_set.remove((u, v))
                
                # Add (x, y)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                edge_set.add((x, y))
                
                # Evaluate
                new_books = compute_books(adj, edge_set)
                new_max = max_book(new_books)
                
                if new_max < current_max:
                    books = new_books
                    current_max = new_max
                    improved = True
                    stuck_counter = 0
                    break
                elif new_max == current_max:
                    # Accept 20% of the time to explore
                    if random.random() < 0.2:
                        books = new_books
                        # current_max unchanged
                        improved = True
                        stuck_counter = 0
                        break
                    else:
                        # Revert
                        adj[x] ^= (1 << y)
                        adj[y] ^= (1 << x)
                        edge_set.remove((x, y))
                        
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add((u, v))
                else:
                    # Revert
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
                    edge_set.remove((x, y))
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
            
            if not improved:
                stuck_counter += 1
                # If stuck for a while, do a kick
                if stuck_counter > 100:
                    # Kick: Random swap
                    kick_edge = random.choice(list(edge_set))
                    u_k, v_k = kick_edge
                    adj[u_k] ^= (1 << v_k)
                    adj[v_k] ^= (1 << u_k)
                    edge_set.remove(kick_edge)
                    
                    while True:
                        x_k = random.randint(0, N - 1)
                        y_k = random.randint(0, N - 1)
                        if x_k == y_k: continue
                        if x_k > y_k: x_k, y_k = y_k, x_k
                        if (x_k, y_k) not in edge_set:
                            break
                    
                    adj[x_k] |= (1 << y_k)
                    adj[y_k] |= (1 << x_k)
                    edge_set.add((x_k, y_k))
                    
                    books = compute_books(adj, edge_set)
                    current_max = max_book(books)
                    stuck_counter = 0
                    
            # Track best state
            if current_max < best_max_in_run:
                best_max_in_run = current_max
                best_state = (adj.copy(), set(edge_set), best_max_in_run)

        return best_state[0], best_state[1], best_state[2]

    best_score = float('inf')
    best_graph = None
    
    # Run multiple restarts
    # We have ~140s. Let's do 5 restarts of ~25s each?
    # Or just 1 long run with kicks?
    # Multiple restarts explores different basins.
    num_restarts = 5
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        adj, edge_set = get_initial_graph()
        adj_final, edge_set_final, score = local_search(adj, edge_set, time.time() + time_per_restart)
        
        if score < best_score:
            best_score = score
            best_graph = list(edge_set_final)
            
    if best_graph is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_graph
