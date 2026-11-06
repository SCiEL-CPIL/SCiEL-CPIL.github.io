"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _score(graph_edges, n=16):
    """Compute the largest triangle degree sum."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in graph_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _tabu_search(n=16, target_edges=65, time_limit=110, seed=None):
    """
    Tabu search to minimize max triangle degree sum.
    Uses edge swap moves with a tabu list.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    
    # Initialize with K_{8,8} + one internal edge
    part_a = list(range(n // 2))
    part_b = list(range(n // 2, n))
    
    # Randomize the partition
    all_vertices = list(range(n))
    random.shuffle(all_vertices)
    part_a = all_vertices[:n // 2]
    part_b = all_vertices[n // 2:]
    
    # Build K_{8,8} edges
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([min(u, v), max(u, v)])
    
    # Add one internal edge within part_a
    random.shuffle(part_a)
    internal_edge = [part_a[0], part_a[1]]
    internal_edge = [min(internal_edge[0], internal_edge[1]), max(internal_edge[0], internal_edge[1])]
    edges.append(internal_edge)
    
    edge_set = set(map(tuple, edges))
    current_score = _score(edges, n)
    best_edges = [e[:] for e in edges]
    best_score = current_score
    
    # Tabu list: store (edge, iteration) for recently moved edges
    tabu_list = {}  # edge -> iteration when it was last added
    tabu_tenure = 10  # keep edges in tabu for 10 iterations
    
    max_iters = 100000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find all possible swaps: remove one edge, add one non-edge
        # To make this efficient, we'll sample a subset
        best_swap_score = current_score
        best_swap = None
        
        # Sample random swaps
        num_samples = 50
        for _ in range(num_samples):
            # Pick random edge to remove
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            removed = (u, v)
            
            # Check if removed edge is tabu (was recently added)
            if removed in tabu_list and it - tabu_list[removed] < tabu_tenure:
                continue
            
            # Find a non-edge to add
            found_add = False
            for _ in range(20):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b:
                    continue
                a, b = min(a, b), max(a, b)
                if (a, b) in edge_set:
                    continue
                if (a, b) == removed:
                    continue
                
                # Check if (a,b) is tabu (was recently removed)
                if (a, b) in tabu_list and it - tabu_list[(a, b)] < tabu_tenure:
                    continue
                
                # Try the swap
                new_edge_set = set(edge_set)
                new_edge_set.remove(removed)
                new_edge_set.add((a, b))
                new_edges = [list(e) for e in new_edge_set]
                
                new_score = _score(new_edges, n)
                
                if new_score < best_swap_score:
                    best_swap_score = new_score
                    best_swap = (removed, (a, b), new_edges)
                    found_add = True
                break
        
        if best_swap is not None:
            removed, added, new_edges = best_swap
            edge_set = set(map(tuple, new_edges))
            edges = new_edges
            current_score = best_swap_score
            
            # Update tabu list
            tabu_list[removed] = it
            tabu_list[added] = it
            
            if current_score < best_score:
                best_score = current_score
                best_edges = [e[:] for e in edges]
        else:
            # No good swap found, try a random move to escape local minimum
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            removed = (u, v)
            
            for _ in range(30):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b:
                    continue
                a, b = min(a, b), max(a, b)
                if (a, b) in edge_set:
                    continue
                if (a, b) == removed:
                    continue
                
                new_edge_set = set(edge_set)
                new_edge_set.remove(removed)
                new_edge_set.add((a, b))
                new_edges = [list(e) for e in new_edge_set]
                
                edge_set = new_edge_set
                edges = new_edges
                current_score = _score(edges, n)
                
                tabu_list[removed] = it
                tabu_list[(a, b)] = it
                break
    
    return best_edges, best_score

def construct():
    """
    Construct a graph on 16 vertices with 65 edges minimizing the max triangle degree sum.
    """
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # Try multiple seeds with tabu search
    for seed in range(5):
        try:
            edges_opt, score_opt = _tabu_search(n, target_edges, time_limit=20, seed=seed)
            if score_opt < best_score:
                best_score = score_opt
                best_edges = [e[:] for e in edges_opt]
        except Exception:
            continue
    
    # Fallback: K_{8,8} + one edge
    if best_edges is None:
        best_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
