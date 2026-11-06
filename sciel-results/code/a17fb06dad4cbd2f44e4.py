"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 145  # safety margin before 150s limit
    
    def compute_books(edge_set):
        """Compute for each edge, the number of triangles containing it."""
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        books = {}
        for u, v in edge_set:
            # Count common neighbors
            common = adj[u] & adj[v]
            books[frozenset((u, v))] = len(common)
        return books
    
    def max_book(edge_set):
        books = compute_books(edge_set)
        return max(books.values()) if books else 0
    
    def get_all_possible_edges():
        """Return all possible edges as frozensets."""
        return [frozenset((i, j)) for i in range(n) for j in range(i + 1, n)]
    
    # Start with K_{20,20} + one intra-partition edge
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add(frozenset((a, b)))
    edges.add(frozenset((0, 1)))
    
    best_edges = set(edges)
    best_score = max_book(best_edges)
    
    # Try different initial intra-partition edges
    for u in range(20):
        for v in range(u + 1, 20):
            test_edges = set()
            for a in range(20):
                for b in range(20, 40):
                    test_edges.add(frozenset((a, b)))
            test_edges.add(frozenset((u, v)))
            score = max_book(test_edges)
            if score < best_score:
                best_score = score
                best_edges = set(test_edges)
    
    # Hill climbing: try to improve by swapping edges
    all_edges = get_all_possible_edges()
    current_edges = set(best_edges)
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try adding a random non-edge and removing a random edge
        non_edges = [e for e in all_edges if e not in current_edges]
        if not non_edges:
            break
        
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            add_edge = random.choice(non_edges)
            remove_edge = random.choice(list(current_edges))
            
            if add_edge == remove_edge:
                continue
            
            new_edges = set(current_edges)
            new_edges.add(add_edge)
            new_edges.discard(remove_edge)
            
            if len(new_edges) != target_edges:
                continue
            
            new_score = max_book(new_edges)
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
                break
            elif new_score == current_score and random.random() < 0.1:
                # Accept equal-score moves with some probability to escape local optima
                current_edges = new_edges
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
    
    # Convert frozensets back to lists
    result = [sorted(e) for e in best_edges]
    return result
# EVOLVE-BLOCK-END
