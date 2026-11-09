"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _compute_max_book(edges, n=40):
    """Compute the maximum book size for the given edges."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_book = 0
    for u, v in edges:
        common = adj[u] & adj[v]
        if len(common) > max_book:
            max_book = len(common)
    return max_book, adj

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0
    
    # 1. Greedy Construction
    # Start with empty graph
    edges = []
    adj = defaultdict(set)
    edge_set = set()
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Pre-compute list of non-edges is dynamic, so we just iterate all_possible_edges
    # and check if in edge_set.
    
    for _ in range(target_edges):
        best_edge = None
        min_book = float('inf')
        candidates = []
        
        # Find non-edges with minimum potential book size
        for u, v in all_possible_edges:
            key = (u, v)
            if key not in edge_set:
                # Calculate book size if we add (u,v)
                # It's the number of common neighbors
                common = adj[u] & adj[v]
                b = len(common)
                
                if b < min_book:
                    min_book = b
                    candidates = [(u, v)]
                elif b == min_book:
                    candidates.append((u, v))
        
        if not candidates:
            break # Should not happen
            
        # Pick randomly from best candidates to break symmetry
        u, v = random.choice(candidates)
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        
    # Check time
    if time.time() - start_time > time_limit - 10:
        return edges

    # 2. Local Search to reduce max book size
    max_book, _ = _compute_max_book(edges, n)
    
    # If we already have 7, we can't do better. 
    # But maybe we can have 6? No, theorem says >= 6.66. So 7 is min.
    # However, if the seed was bad, getting 7 is a huge win.
    # Let's try to improve if max_book > 7.
    
    if max_book <= 7:
        return edges

    best_edges = list(edges)
    best_max_book = max_book
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    current_edges = list(edges)
    current_adj = {k: set(v) for k, v in adj.items()}
    current_edge_set = set(edge_set)
    current_max_book = max_book
    
    # Precompute book sizes for all current edges for faster access?
    # We will recompute max book periodically or use a heap.
    # For simplicity, recompute max book every 100 iterations or when a change is accepted.
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        if iterations % 10000 == 0:
            # Update best
            pass

        # Pick a random edge to remove
        idx = random.randint(0, len(current_edges) - 1)
        u1, v1 = current_edges[idx]
        rem_key = (min(u1, v1), max(u1, v1))
        
        # Pick a random non-edge to add
        while True:
            add_u, add_v = random.choice(all_possible_edges)
            add_key = (min(add_u, add_v), max(add_u, add_v))
            if add_key not in current_edge_set:
                break
                
        # Apply swap
        current_edge_set.remove(rem_key)
        current_adj[u1].remove(v1)
        current_adj[v1].remove(u1)
        
        current_edge_set.add(add_key)
        current_adj[add_u].add(add_v)
        current_adj[add_v].add(add_u)
        
        current_edges[idx] = [add_u, add_v]
        
        # Calculate new max book
        # Only recompute if necessary? 
        # Let's just recompute max book for all 401 edges. 
        # 401 * (intersection of sets) 
        # Intersection is fast if sets are small.
        new_max_book = 0
        for u, v in current_edges:
            common = current_adj[u] & current_adj[v]
            if len(common) > new_max_book:
                new_max_book = len(common)
                
        # Acceptance
        delta = new_max_book - current_max_book
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < pow(2.718, -delta / T)):
            current_max_book = new_max_book
            if new_max_book < best_max_book:
                best_max_book = new_max_book
                best_edges = list(current_edges)
                # Early stop if we hit the theoretical minimum
                if best_max_book <= 7:
                    break
        else:
            # Revert
            current_edge_set.remove(add_key)
            current_edge_set.add(rem_key)
            
            current_adj[add_u].remove(add_v)
            current_adj[add_v].remove(add_u)
            current_adj[u1].add(v1)
            current_adj[v1].add(u1)
            
            current_edges[idx] = [u1, v1]
            
        T = max(T * cooling_rate, T_min)
        
    return best_edges

# EVOLVE-BLOCK-END
