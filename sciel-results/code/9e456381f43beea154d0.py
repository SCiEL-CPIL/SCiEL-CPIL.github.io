import random
import time

def construct():
    n = 40
    target_edges = 401
    
    # Initial construction:
    # A = 0..7, B = 8..19, C = 20..26, D = 27..39
    # Edges:
    # A clique
    # A-C complete
    # B-C complete
    # B-D complete
    # Plus 77 random A-D edges
    
    edges = set()
    
    # A clique
    for i in range(8):
        for j in range(i+1, 8):
            edges.add((i, j))
            
    # A-C complete
    for i in range(8):
        for j in range(20, 27):
            edges.add((i, j))
            
    # B-C complete
    for i in range(8, 20):
        for j in range(20, 27):
            edges.add((i, j))
            
    # B-D complete
    for i in range(8, 20):
        for j in range(27, 40):
            edges.add((i, j))
            
    # Current count: 28 + 56 + 84 + 156 = 324
    # Need 77 more. Add random A-D edges.
    possible_ad = [(i, j) for i in range(8) for j in range(27, 40)]
    random.shuffle(possible_ad)
    for e in possible_ad[:77]:
        edges.add(e)
        
    edges = list(edges)
    
    # Local Search
    def compute_book_sizes(edge_list):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
            
        # Compute book size for each edge
        book_sizes = {}
        for u, v in edge_set:
            common = adj[u].intersection(adj[v])
            book_sizes[(u,v)] = len(common)
        return book_sizes, edge_set

    def get_max_book(edge_list):
        book_sizes, _ = compute_book_sizes(edge_list)
        if not book_sizes:
            return 0
        return max(book_sizes.values())

    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    current_edges = edges
    current_max = get_max_book(current_edges)
    
    while time.time() < deadline:
        # Pick a random edge to remove and a random non-edge to add
        u, v = random.choice(current_edges)
        # Find a non-edge
        # To speed up, pick random pair
        attempts = 0
        while attempts < 10:
            a, b = random.randint(0, n-1), random.randint(0, n-1)
            if a == b: continue
            key = (min(a,b), max(a,b))
            if key not in {(min(x,y), max(x,y)) for x,y in current_edges}:
                break
            attempts += 1
        else:
            continue
            
        # Swap
        new_edges = [e for e in current_edges if e != [u,v]]
        new_edges.append([a,b])
        
        new_max = get_max_book(new_edges)
        
        if new_max <= current_max:
            current_edges = new_edges
            current_max = new_max
            
    return [[u, v] for u, v in current_edges]
