import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    # Initialize with a random graph of 401 edges
    # Using a simple random generation
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    current_edges = set(all_pairs[:target_edges])
    
    # Helper to count triangles containing an edge
    # For speed, we can maintain a dictionary of adjacency or just compute on the fly for small n
    # n=40, edges=401. Computing common neighbors for an edge (u,v) is O(deg(u)+deg(v))
    # Let's use a set of edges for O(1) lookup.
    
    def get_score(edges):
        max_book = 0
        # To make this faster, we can precompute degrees or just iterate
        # 401 edges * O(40) common neighbor check = ~16000 ops. Very fast.
        # But we need the max.
        for u, v in edges:
            # Count common neighbors
            # Since n is small, we can just check all w in 0..39
            # Or better, iterate over neighbors of u and check if connected to v
            # Let's build an adjacency list once per score calculation? No, that's expensive if done every step.
            # However, 16000 ops is negligible.
            # Let's optimize:
            # common = 0
            # for w in range(n):
            #     if (u, w) in edges and (v, w) in edges: # assuming u<w or similar normalization
            #         common += 1
            # 
            # Actually, let's normalize edges to (min, max)
            pass
        
        # A better way for the loop:
        # Precompute nothing, just check.
        # But we need to return the max.
        
        # Let's try a slightly more efficient local score update strategy later.
        # For now, full re-evaluation is O(E * N) = 401 * 40 = 16040 checks.
        # In Python, 16k checks is ~1ms. 110s / 1ms = 110,000 steps.
        # This is a decent number of steps for local search.
        
        max_b = 0
        edge_list = list(edges)
        for u, v in edge_list:
            # Ensure u < v for consistency if we used sets, but here we just check existence
            count = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in edges and e2 in edges:
                    count += 1
            if count > max_b:
                max_b = count
                if max_b == 0: # Can't get better
                    break
        return max_b

    current_score = get_score(current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing / Local Search
    # Move: Swap an edge. Remove one, add one.
    # Or: Flip an edge. Add if not present, remove if present.
    # Flipping is simpler.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to flip
        # To ensure we stay near the target edge count, we can bias towards flips that don't change count?
        # No, flipping changes count by +/- 1.
        # We need exactly 401 edges.
        # So we should do: Remove one existing, Add one non-existing.
        # This keeps count constant.
        
        if len(current_edges) != target_edges:
            # Repair
            if len(current_edges) < target_edges:
                # Add random missing
                # This is slow to find missing. 
                # Better to always do swap.
                pass
            # Let's stick to swaps to maintain count.
            # Pick random existing edge to remove
            # Pick random non-existing edge to add
            
        # Generate a swap
        # Sample random u, v for new edge
        # Sample random existing edge to remove
        
        # Efficient sampling of non-edge:
        # Since 401/780 is ~50%, random sampling might hit existing edges.
        # Let's just sample pairs until we find a non-edge.
        
        new_u, new_v = random.sample(vertices, 2)
        new_edge = (new_u, new_v) if new_u < new_v else (new_v, new_u)
        
        # If it's already there, try again or pick another
        attempts = 0
        while new_edge in current_edges and attempts < 10:
            new_u, new_v = random.sample(vertices, 2)
            new_edge = (new_u, new_v) if new_u < new_v else (new_v, new_u)
            attempts += 1
            
        if new_edge in current_edges:
            continue
            
        # Remove a random existing edge
        old_edge = random.choice(list(current_edges))
        
        # Apply move
        current_edges.remove(old_edge)
        current_edges.add(new_edge)
        
        new_score = get_score(current_edges)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Revert
            current_edges.remove(new_edge)
            current_edges.add(old_edge)
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Return best found
    # Format: list of [u, v]
    return [[u, v] for u, v in best_edges]
