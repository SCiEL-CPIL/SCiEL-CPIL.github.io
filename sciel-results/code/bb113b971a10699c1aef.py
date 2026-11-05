import random
import time
import sys

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    max_time = 110.0  # Safety margin

    # Initialize with a random graph
    # To be closer to the seed, we could start with K_{8,8} + 1, but random might be better for exploration.
    # Let's start with a random graph.
    
    def random_graph():
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        return set(all_edges[:target_edges])
    
    edges = random_graph()
    
    def get_score(edge_set):
        # Build adjacency list and degree
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        max_sum = 0
        # Find all triangles
        # For each pair (u, v) that is an edge, check common neighbors
        # Or iterate over all triples
        # n=16 is small, so O(n^3) is fine.
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    for w in range(v+1, n):
                        if w in adj[u] and w in adj[v]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    best_score = get_score(edges)
    best_edges = set(edges)
    
    # Local search
    while time.time() - start_time < max_time:
        current_score = get_score(edges)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        if best_score <= 24:
            break # Found a very good solution
            
        # Find the worst triangle
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        worst_tri = None
        max_s = 0
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    for w in range(v+1, n):
                        if w in adj[u] and w in adj[v]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_s:
                                max_s = s
                                worst_tri = (u, v, w)
        
        if not worst_tri:
            break
            
        u, v, w = worst_tri
        
        # Try to swap an edge incident to u, v, or w
        # with a non-edge
        improved = False
        candidates = []
        
        # Collect edges to remove
        remove_candidates = []
        for vert in (u, v, w):
            for neighbor in adj[vert]:
                # Ensure we don't duplicate
                if vert < neighbor:
                    remove_candidates.append((vert, neighbor))
                else:
                    remove_candidates.append((neighbor, vert))
        
        # Collect non-edges to add
        # We want to add an edge (x, y) that is not in edges
        # And ideally doesn't create a new triangle with high sum
        # Let's just pick a few random non-edges
        
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges and (j, i) not in edges:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        # Limit non-edges to check for speed
        non_edges_to_check = non_edges[:50]
        
        for rem_edge in remove_candidates:
            for add_edge in non_edges_to_check:
                # Check if add_edge is same as rem_edge
                if set(rem_edge) == set(add_edge):
                    continue
                
                # Apply swap
                new_edges = set(edges)
                new_edges.remove(rem_edge)
                new_edges.add(add_edge)
                
                # Check if new_edges is valid (it is, size is same)
                new_score = get_score(new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a random swap to escape local minimum
            # Pick a random edge to remove
            rem_edge = random.choice(list(edges))
            # Pick a random non-edge to add
            if non_edges:
                add_edge = random.choice(non_edges)
                if set(rem_edge) != set(add_edge):
                    new_edges = set(edges)
                    new_edges.remove(rem_edge)
                    new_edges.add(add_edge)
                    # Only accept if it doesn't increase score too much? Or just accept with probability?
                    # For simplicity, accept if score is same or better, or just accept to explore.
                    # Let's accept if score <= best_score + 2 (simulated annealing style, but simple)
                    new_score = get_score(new_edges)
                    if new_score <= best_score + 2:
                        edges = new_edges

    return {'n': n, 'edges': [list(e) for e in best_edges]}
