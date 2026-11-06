import time
import random
import itertools

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 110  # Safety margin for 120s limit

    # Initial graph: K_{8,8} + one edge (0,1)
    # Part 0: 0..7, Part 1: 8..15
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            edges_set.add((min(u, v), max(u, v)))
    edges_set.add((0, 1))
    
    def get_edges_list(es):
        return [[u, v] for u, v in es]

    def calc_score(es):
        # Build adjacency and degrees
        adj = [[0]*n for _ in range(n)]
        deg = [0]*n
        for u, v in es:
            adj[u][v] = 1
            adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        # Find all triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = calc_score(edges_set)
    best_score = current_score
    best_edges = set(edges_set)

    # Local Search
    # Move: remove edge (u,v), add edge (x,y)
    # To be efficient, we can precompute non-edges.
    
    while time.time() < deadline:
        improved = False
        # Randomize to escape local minima
        if random.random() < 0.1:
            # Perturb
            es = set(best_edges)
            for _ in range(5):
                if len(es) > 0:
                    # Remove random edge
                    rem_edge = random.choice(list(es))
                    es.remove(rem_edge)
                    # Add random non-edge
                    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (min(i,j), max(i,j)) not in es]
                    if non_edges:
                        add_edge = random.choice(non_edges)
                        es.add(add_edge)
            current_score = calc_score(es)
            if current_score < best_score:
                best_score = current_score
                best_edges = es
                improved = True
                current_edges = es
            else:
                current_edges = set(best_edges)
        else:
            current_edges = set(best_edges)
        
        # Hill climbing step
        # Try all possible swaps? 65 * (120-65) ~ 3500 swaps. Fast enough.
        # Or random sample of swaps.
        
        # Let's try all swaps for a few iterations, then random.
        # Since n=16, total pairs is 120.
        # Edges: 65, Non-edges: 55.
        # Swaps: 65 * 55 = 3575.
        
        found_improvement = False
        # Iterate over edges
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        for e1 in edge_list:
            if time.time() > deadline:
                break
            u, v = e1
            # Find non-edges to add
            # We can iterate through all pairs (x,y) not in current_edges and not (u,v)
            # Optimization: precompute non-edges
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (min(i,j), max(i,j)) not in current_edges and (i,j) != e1]
            
            for e2 in non_edges:
                new_edges = set(current_edges)
                new_edges.remove(e1)
                new_edges.add(e2)
                
                # Quick check: if e1 and e2 don't affect triangles involving high degree nodes, maybe skip?
                # Just calculate score.
                s = calc_score(new_edges)
                if s < current_score:
                    current_score = s
                    current_edges = new_edges
                    found_improvement = True
                    break # Restart loop with new current_edges
            if found_improvement:
                break
        
        if found_improvement:
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
            improved = True
        else:
            # No improvement found in this pass
            # Try random restart or perturbation
            if random.random() < 0.5:
                # Do a perturbation
                es = set(best_edges)
                for _ in range(10):
                    if len(es) > 0:
                        rem_edge = random.choice(list(es))
                        es.remove(rem_edge)
                        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (min(i,j), max(i,j)) not in es]
                        if non_edges:
                            add_edge = random.choice(non_edges)
                            es.add(add_edge)
                current_score = calc_score(es)
                if current_score < best_score:
                    best_score = current_score
                    best_edges = es
                    improved = True
                else:
                    # Keep best, try again next loop
                    pass
            else:
                # Break if stuck? No, keep trying random swaps
                pass

    return {'n': n, 'edges': get_edges_list(best_edges)}
