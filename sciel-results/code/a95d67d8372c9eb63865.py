import itertools
import random
import time

def construct():
    n = 9
    # Generate all 5-uniform edges on n vertices
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Convert to bitmasks for fast checking
    all_edges_bitmasks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        all_edges_bitmasks.append(mask)
        
    def is_2_colorable(edge_bitmasks):
        # Check if there exists a coloring c (0..2^n - 1)
        # such that no edge is monochromatic.
        # Edge e is monochromatic if (e & c) == e or (e & c) == 0
        # We can early exit if we find a valid coloring.
        for c in range(1 << n):
            valid = True
            for e in edge_bitmasks:
                if (e & c) == e or (e & c) == 0:
                    valid = False
                    break
            if valid:
                return True
        return False

    best_edges = all_edges
    best_count = len(all_edges)
    
    # Time limit guard
    start_time = time.time()
    timeout = 110  # Safety margin
    
    while time.time() - start_time < timeout:
        # Shuffle edges to get different removal orders
        shuffled_indices = list(range(len(all_edges)))
        random.shuffle(shuffled_indices)
        
        current_edges = set(all_edges_bitmasks)
        
        # Greedy removal
        for idx in shuffled_indices:
            if time.time() - start_time > timeout:
                break
                
            e = all_edges_bitmasks[idx]
            if e not in current_edges:
                continue
                
            # Try removing e
            test_edges = current_edges - {e}
            
            # Check if remaining hypergraph is 2-colorable
            if is_2_colorable(list(test_edges)):
                # If it is 2-colorable, we cannot remove e
                pass
            else:
                # If it is not 2-colorable, we can remove e
                current_edges = test_edges
                
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            # Reconstruct the list of edges from bitmasks
            best_edges = []
            for mask in current_edges:
                edge = []
                for v in range(n):
                    if mask & (1 << v):
                        edge.append(v)
                best_edges.append(edge)
                
    return {'n': n, 'edges': best_edges}
