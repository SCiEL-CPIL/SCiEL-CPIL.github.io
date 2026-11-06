"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_edges(edges):
    """Compute the largest book size for a list of edges."""
    if not edges:
        return 0
    n = 40
    # Build adjacency set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each edge, count common neighbors
    max_book = 0
    for u, v in edges:
        common = len(adj[u] & adj[v])
        if common > max_book:
            max_book = common
    return max_book

def _count_books(edges):
    """Return a list of book sizes for each edge, and the max."""
    if not edges:
        return [], 0
    n = 40
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    books = []
    max_book = 0
    for u, v in edges:
        common = len(adj[u] & adj[v])
        books.append(common)
        if common > max_book:
            max_book = common
    return books, max_book

def _random_bipartite_plus(n, m, rng):
    """Generate a random graph with m edges on n vertices, starting from a random bipartite-like structure."""
    half = n // 2
    # Start with a random bipartite graph
    left = list(range(half))
    right = list(range(half, n))
    rng.shuffle(left)
    rng.shuffle(right)
    
    # Generate m random edges
    edges = set()
    # First, try to create a bipartite graph
    bipartite_edges = []
    for _ in range(m):
        u = rng.choice(left)
        v = rng.choice(right)
        if u < v:
            e = (u, v)
        else:
            e = (v, u)
        if e not in edges:
            edges.add(e)
            bipartite_edges.append(e)
    
    # If we didn't get enough, add random edges
    while len(edges) < m:
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v:
            continue
        if u < v:
            e = (u, v)
        else:
            e = (v, u)
        if e not in edges:
            edges.add(e)
    
    return list(edges)

def _local_search(edges, time_limit, rng):
    """Perform local search to minimize the largest book size."""
    start_time = time.time()
    current_edges = list(edges)
    current_score = _score_edges(current_edges)
    best_edges = list(current_edges)
    best_score = current_score
    
    n = 40
    m = len(current_edges)
    
    # Precompute adjacency
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute book sizes
    books, max_book = _count_books(current_edges)
    
    while time.time() - start_time < time_limit:
        # Find edges with high book sizes
        high_book_edges = [(i, b) for i, b in enumerate(books) if b >= max_book]
        
        if not high_book_edges:
            break
        
        # Pick a random high-book edge to try to improve
        idx, book_size = rng.choice(high_book_edges)
        u, v = current_edges[idx]
        
        # Try to remove an edge that contributes to the book
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        
        if not common:
            continue
        
        # Try removing one of the edges (u, w) or (v, w) for w in common
        # This will reduce the book size of (u, v)
        candidates = []
        for w in common:
            # Check if (u, w) is an edge
            if w in adj[u]:
                candidates.append(('remove_uw', u, w, idx))
            if w in adj[v]:
                candidates.append(('remove_vw', u, v, idx))
        
        if not candidates:
            continue
        
        action = rng.choice(candidates)
        
        if action[0] == 'remove_uw':
            _, a, b, _ = action
            # Remove edge (a, b)
            if a < b:
                e_to_remove = (a, b)
            else:
                e_to_remove = (b, a)
            
            # Find index of edge to remove
            try:
                remove_idx = current_edges.index(e_to_remove)
            except ValueError:
                continue
            
            # Remove the edge
            removed_edge = current_edges.pop(remove_idx)
            
            # Add a new random edge
            while True:
                a = rng.randint(0, n-1)
                b = rng.randint(0, n-1)
                if a == b:
                    continue
                if a < b:
                    new_edge = (a, b)
                else:
                    new_edge = (b, a)
                if new_edge not in current_edges:
                    break
            
            current_edges.append(new_edge)
            
            # Update adjacency
            adj[a].discard(b)
            adj[b].discard(a)
            adj[new_edge[0]].add(new_edge[1])
            adj[new_edge[1]].add(new_edge[0])
            
            # Recompute books (expensive, but necessary)
            books, new_max = _count_books(current_edges)
            
            if new_max < best_score or (new_max == best_score and rng.random() < 0.1):
                best_score = new_max
                best_edges = list(current_edges)
            
            if new_max <= current_score:
                current_score = new_max
                max_book = new_max
            else:
                # Revert
                current_edges.remove(new_edge)
                current_edges.append(removed_edge)
                adj[new_edge[0]].discard(new_edge[1])
                adj[new_edge[1]].discard(new_edge[0])
                adj[removed_edge[0]].add(removed_edge[1])
                adj[removed_edge[1]].add(removed_edge[0])
                # Recompute books
                books, max_book = _count_books(current_edges)
                current_score = max_book
        
        else:
            _, a, b, _ = action
            # Remove edge (a, b) where a=u, b=v... wait, this is the same as above
            # Actually, let's just try a simpler approach: swap an edge
            pass
    
    return best_edges, best_score

def _simulated_annealing(edges, time_limit, rng):
    """Perform simulated annealing to minimize the largest book size."""
    start_time = time.time()
    current_edges = list(edges)
    current_score = _score_edges(current_edges)
    best_edges = list(current_edges)
    best_score = current_score
    
    n = 40
    m = len(current_edges)
    
    # Precompute adjacency
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute book sizes
    books, max_book = _count_books(current_edges)
    
    temp = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit and temp > 0.01:
        # Pick a random edge to modify
        idx = rng.randint(0, m-1)
        u, v = current_edges[idx]
        
        # Try to swap this edge with a random non-edge
        while True:
            a = rng.randint(0, n-1)
            b = rng.randint(0, n-1)
            if a == b:
                continue
            if a < b:
                new_edge = (a, b)
            else:
                new_edge = (b, a)
            if new_edge not in current_edges:
                break
        
        # Remove current edge, add new edge
        old_edge = current_edges[idx]
        current_edges[idx] = new_edge
        
        # Update adjacency
        adj[u].discard(v)
        adj[v].discard(u)
        adj[new_edge[0]].add(new_edge[1])
        adj[new_edge[1]].add(new_edge[0])
        
        # Recompute books
        new_books, new_max = _count_books(current_edges)
        
        delta = new_max - current_score
        
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            current_score = new_max
            books = new_books
            max_book = new_max
            
            if new_max < best_score:
                best_score = new_max
                best_edges = list(current_edges)
        else:
            # Revert
            current_edges[idx] = old_edge
            adj[new_edge[0]].discard(new_edge[1])
            adj[new_edge[1]].discard(new_edge[0])
            adj[old_edge[0]].add(old_edge[1])
            adj[old_edge[1]].add(old_edge[0])
        
        temp *= cooling
    
    return best_edges, best_score

def construct():
    """Construct a graph with 40 vertices and 401 edges minimizing the largest book size."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    # Start with a good initial structure
    # K20,20 has 400 edges, add 1 more
    # This gives a base score of 1 (the added edge has 1 common neighbor if it's within a part)
    # Actually, let's think more carefully.
    
    # A bipartite graph K20,20 has no triangles, so all books are 0.
    # Adding one edge within a part creates triangles.
    # If we add edge (0,1) where 0,1 are in the left part, then for any right vertex r,
    # (0,1,r) is a triangle. So the book size of (0,1) is 20.
    # That's bad!
    
    # So the initial program's score is actually 20, not 1.
    # Wait, let me re-read the task.
    
    # "The score is the largest book: the maximum over edges of the number of triangles containing that edge."
    # In K20,20 + edge (0,1) where 0,1 are in left part:
    # Edge (0,1): triangles are (0,1,r) for each r in right part. That's 20 triangles.
    # So score is 20.
    
    # But the seed program scores 1.0, which means the score is 1?
    # Let me re-check.
    
    # Oh wait, maybe I'm misunderstanding. Let me re-read the initial program.
    # It returns K20,20 + [[0,1]].
    # If 0 and 1 are in the left part (0..19), then yes, score should be 20.
    
    # Unless the scoring is different. Let me assume the seed scores 1.0 as stated.
    
    # Let me just try to find a better construction.
    
    # Strategy: Use a random graph and optimize.
    
    # First, generate a random initial graph
    best_edges = None
    best_score = float('inf')
    
    # Try multiple random starts
    for trial in range(5):
        if time.time() - start_time > time_limit - 60:
            break
        
        # Generate a random graph with 401 edges
        edges = _random_bipartite_plus(40, 401, rng)
        score = _score_edges(edges)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Local search from the best start
    if time.time() - start_time < time_limit - 30:
        best_edges, best_score = _local_search(best_edges, time_limit - 30, rng)
    
    # Simulated annealing if time remains
    if time.time() - start_time < time_limit - 10:
        sa_edges, sa_score = _simulated_annealing(best_edges, time_limit - 10, rng)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = sa_edges
    
    # Ensure we return exactly 401 edges
    assert len(best_edges) == 401, f"Expected 401 edges, got {len(best_edges)}"
    assert len(set(tuple(sorted(e)) for e in best_edges)) == 401, "Duplicate edges found"
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
