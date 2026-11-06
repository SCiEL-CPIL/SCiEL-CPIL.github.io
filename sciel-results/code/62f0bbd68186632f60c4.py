import random
import time

def construct():
    n = 40
    m = 401
    
    # Initialize with K20,20 + 1 edge
    # A = 0..19, B = 20..39
    edges = set()
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
    edges.add((0, 1))
    
    # Helper to get neighbors
    def get_neighbors(edge_set):
        neighbors = {i: set() for i in range(n)}
        for u, v in edge_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors

    def get_max_book(edge_set):
        neighbors = get_neighbors(edge_set)
        max_book = 0
        for u, v in edge_set:
            common = len(neighbors[u] & neighbors[v])
            if common > max_book:
                max_book = common
        return max_book

    current_edges = edges
    current_max_book = get_max_book(current_edges)
    
    # Local Search
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = current_edges
    best_max_book = current_max_book
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # To speed up, we might need to update book sizes incrementally, 
    # but for simplicity and robustness within time limit, 
    # we can do a few thousand iterations of full recalculation 
    # or incremental updates if we optimize the neighbor lookup.
    
    # Given n=40, m=401, full recalc is O(m * deg) which is small.
    # 401 * 20 ~ 8000 ops per eval.
    # 10,000 iterations ~ 80M ops. Python might be slow, but 110s is generous.
    
    while time.time() < deadline:
        # Random move: Remove one edge, add one edge
        # Ensure we don't add an existing edge
        if not current_edges:
            break
            
        # Pick a random edge to remove
        edge_to_remove = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        # To avoid O(m) check for non-edge, we can pick random u, v and check
        # Or maintain a list of non-edges.
        # For simplicity, pick random u, v
        added = False
        for _ in range(100): # Try 100 times to find a non-edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            # Check if it's the same as removed edge
            if (u, v) == edge_to_remove or (v, u) == edge_to_remove:
                continue
            new_edges = set(current_edges)
            new_edges.remove(edge_to_remove)
            new_edges.add((u, v))
            
            # Calculate new max book
            new_max_book = get_max_book(new_edges)
            
            if new_max_book < best_max_book:
                best_max_book = new_max_book
                best_edges = new_edges
                current_edges = new_edges
                current_max_book = new_max_book
            elif new_max_book < current_max_book:
                # Accept improvement
                current_edges = new_edges
                current_max_book = new_max_book
            elif random.random() < 0.1:
                # Accept worse with low probability (annealing)
                current_edges = new_edges
                current_max_book = new_max_book
                
            added = True
            break
            
    # Convert best_edges to list of lists
    return [list(e) for e in best_edges]
