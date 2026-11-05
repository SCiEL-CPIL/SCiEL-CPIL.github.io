import random
import time

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leave 10s margin
    
    best_edges = None
    best_score = n  # Max possible is n
    
    # Helper to check K4-free and estimate triangle-free set size
    def is_k4_free(adj):
        # adj is a list of sets
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    # Check if any common neighbor forms a K4
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            # Check if i, j, k form a triangle with a 4th vertex?
                            # Actually, just check if adj[i] & adj[j] & adj[k] is non-empty
                            # If there is a 4th vertex l adjacent to i, j, k, then i,j,k,l is K4.
                            # So if adj[i] & adj[j] & adj[k] is not empty, we have a K4.
                            if adj[i] & adj[j] & adj[k]:
                                return False
        return True

    def max_triangle_free_size(adj):
        # Greedy approximation: 
        # A triangle-free set is a set of vertices with no triangles.
        # This is not the same as independent set.
        # We can use a simple heuristic: 
        # Try to find a large subset with no triangles.
        # This is NP-hard. 
        # For a rough estimate, we can count the number of triangles in the graph.
        # If there are many triangles, the triangle-free sets are likely smaller.
        # Let's just return a fixed value or a simple metric for now, 
        # and rely on the evaluator for the exact score.
        # But we need to compare candidates.
        # A better metric: The number of triangles in the graph.
        # More triangles -> harder to find large triangle-free sets.
        triangles = 0
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    triangles += len(adj[i] & adj[j])
        return -triangles  # We want to maximize triangles to minimize triangle-free set
    
    # Random restart
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Generate random graph
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Remove K4s
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if j in adj[i]:
                        common = adj[i] & adj[j]
                        for k in common:
                            if k > j:
                                if adj[i] & adj[j] & adj[k]:
                                    # Found K4: i, j, k, l (where l is in the intersection)
                                    # Remove an edge, e.g., (i, j)
                                    adj[i].remove(j)
                                    adj[j].remove(i)
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        if not is_k4_free(adj):
            continue
            
        score = max_triangle_free_size(adj)
        if score > best_score: # Maximizing negative triangles
            best_score = score
            # Convert adj to list of edges
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            best_edges = edges
            
    if best_edges is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return best_edges
