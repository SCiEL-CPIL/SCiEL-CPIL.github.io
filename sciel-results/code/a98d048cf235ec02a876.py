import time
import random

def construct():
    n = 16
    m = 65
    
    # Initial graph: K_{8,8} + (0,1)
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = set(all_edges) - edges
    
    # Precompute all triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    def get_score(edges_set):
        # Compute degrees
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
            
        # Create adjacency set for fast lookup
        adj = set()
        for u, v in edges_set:
            adj.add((u, v))
            adj.add((v, u))
            
        max_sum = 0
        for u, v, w in triples:
            if (u, v) in adj and (v, w) in adj and (u, w) in adj:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
        return max_sum

    current_score = get_score(edges)
    best_score = current_score
    best_edges = set(edges)
    
    start_time = time.time()
    timeout = 140.0 # Leave margin
    
    edge_list = list(edges)
    non_edge_list = list(non_edges)
    
    while time.time() - start_time < timeout:
        # Pick a random edge to remove
        edge_to_remove = random.choice(edge_list)
        
        # Pick a random non-edge to add
        edge_to_add = random.choice(non_edge_list)
        
        # Perform swap
        new_edges = set(edges)
        new_edges.remove(edge_to_remove)
        new_edges.add(edge_to_add)
        
        new_score = get_score(new_edges)
        
        if new_score < current_score:
            edges = new_edges
            edge_list = list(edges)
            
            # Update non_edges
            non_edges.discard(edge_to_add)
            non_edges.add(edge_to_remove)
            non_edge_list = list(non_edges)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
        elif new_score == current_score:
            # Accept to explore
            if random.random() < 0.5:
                edges = new_edges
                edge_list = list(edges)
                
                non_edges.discard(edge_to_add)
                non_edges.add(edge_to_remove)
                non_edge_list = list(non_edges)
                
                current_score = new_score
        # else reject

    # Convert back to list of lists
    final_edges = [[u, v] for u, v in best_edges]
    return {'n': n, 'edges': final_edges}
