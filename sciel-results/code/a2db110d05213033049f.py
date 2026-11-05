import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    N = 40
    M = 401
    
    best_graph = None
    best_score = float('inf')
    
    def get_book_sizes(adj):
        books = {}
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    books[(u, v)] = common.bit_count()
        return books

    def max_book(books):
        if not books:
            return 0
        return max(books.values())

    def random_graph():
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        edges = all_edges[:M]
        
        adj = [0] * N
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj, edges

    def local_search(adj, edges, time_limit):
        books = get_book_sizes(adj)
        current_max = max_book(books)
        
        edge_set = set(edges)
        
        while time.time() < time_limit:
            if current_max <= 7:
                break
                
            max_edges = [e for e, b in books.items() if b == current_max]
            if not max_edges:
                break
            e_max = random.choice(max_edges)
            u, v = e_max
            
            # Remove e_max
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edge_set.remove(e_max)
            
            # Find best candidate to add
            best_candidate = None
            min_common = float('inf')
            
            # Try a few random candidates
            for _ in range(20):
                x = random.randint(0, N - 1)
                y = random.randint(0, N - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) in edge_set:
                    continue
                
                common = (adj[x] & adj[y]).bit_count()
                if common < min_common:
                    min_common = common
                    best_candidate = (x, y)
            
            if best_candidate:
                x, y = best_candidate
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                edge_set.add((x, y))
                
                new_books = get_book_sizes(adj)
                new_max = max_book(new_books)
                
                if new_max < current_max:
                    books = new_books
                    current_max = new_max
                else:
                    # Revert
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
                    edge_set.remove((x, y))
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add(e_max)
            else:
                # Revert if no candidate found
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add(e_max)
            
        return list(edge_set), current_max

    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        adj, edges = random_graph()
        final_edges, score = local_search(adj, edges, deadline - 5)
        
        if score < best_score:
            best_score = score
            best_graph = final_edges
            
    if best_graph is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_graph
