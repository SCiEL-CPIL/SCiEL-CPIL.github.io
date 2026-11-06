import itertools
import random
import time
import sys

def construct():
    # Best known minimum is 51 edges.
    # We aim to find a non-2-colorable 5-uniform hypergraph with <= 51 edges.
    # We will use local search starting from random candidates.
    
    start_time = time.time()
    timeout = 115  # Safety margin
    
    def is_not_2_colorable(edges, n):
        """
        Check if the hypergraph is NOT 2-colorable.
        Uses backtracking with unit propagation.
        edges: list of tuples of vertices (0..n-1)
        n: number of vertices
        Returns True if NOT 2-colorable (i.e., no proper 2-coloring exists).
        """
        # Precompute adjacency/incidence for faster propagation
        # For each vertex, list of edges containing it
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        # State: color of each vertex. -1 is uncolored, 0/1 are colors.
        colors = [-1] * n
        # Queue for unit propagation
        # We maintain counts of 0s and 1s for each edge
        edge_counts = [[0, 0] for _ in range(len(edges))]
        
        # Initialize counts
        for i, e in enumerate(edges):
            for v in e:
                # All are -1 initially, so counts are 0
                pass
        
        def propagate(vertex, color):
            """
            Assign color to vertex and propagate implications.
            Returns False if conflict detected.
            """
            if colors[vertex] != -1:
                if colors[vertex] != color:
                    return False
                else:
                    return True
            
            colors[vertex] = color
            
            # Update counts for all edges containing this vertex
            for e_idx in vertex_edges[vertex]:
                edge_counts[e_idx][color] += 1
                # Check for monochromatic edge
                if edge_counts[e_idx][color] == 5:
                    return False
                
                # Unit propagation: if 4 are one color, the 5th must be the other
                # We need to find the uncolored vertex in this edge
                # This is O(5) per edge
                uncolored = -1
                uncolored_count = 0
                for v in edges[e_idx]:
                    if colors[v] == -1:
                        uncolored_count += 1
                        uncolored = v
                
                if uncolored_count == 1:
                    # Determine required color
                    if edge_counts[e_idx][0] == 4:
                        required = 1
                    elif edge_counts[e_idx][1] == 4:
                        required = 0
                    else:
                        # Should not happen if we only propagate when 4 are fixed
                        # But if 4 are 0s, 1 must be 1. If 4 are 1s, 1 must be 0.
                        # What if 3 are 0 and 2 are 1? No propagation.
                        # What if 4 are 0 and 1 is uncolored? Propagate 1.
                        # What if 4 are 1 and 1 is uncolored? Propagate 0.
                        # If counts are e.g. 3,1,1(uncolored), no unit clause.
                        return True # No immediate propagation from this edge state
                
                    # Recursively assign
                    if not propagate(uncolored, required):
                        return False
            return True
        
        # Backtracking search
        def backtrack(v_idx):
            if v_idx == n:
                return True # All vertices colored, no conflict found
            
            # Try color 0
            # Save state
            old_colors = colors.copy()
            old_counts = [c[:] for c in edge_counts]
            
            if propagate(v_idx, 0):
                if backtrack(v_idx + 1):
                    return True
            
            # Restore state
            colors[:] = old_colors
            for i in range(len(edge_counts)):
                edge_counts[i][0] = old_counts[i][0]
                edge_counts[i][1] = old_counts[i][1]
            
            # Try color 1
            old_colors = colors.copy()
            old_counts = [c[:] for c in edge_counts]
            
            if propagate(v_idx, 1):
                if backtrack(v_idx + 1):
                    return True
            
            return False
        
        # Heuristic: order vertices by degree?
        # For now, simple order.
        # To make it faster, we can pick the vertex with the most constraints first.
        # But for n=15, simple order is likely fine.
        
        # Optimization: If any edge is monochromatic in initial state (impossible here as all -1)
        
        # Run backtracking
        # Note: The above propagate function modifies state globally.
        # The backtracking logic above restores state manually.
        
        # A simpler, potentially faster approach for small n:
        # Iterate over all 2^n colorings? 2^15 = 32768.
        # Checking each coloring takes O(E * 5). 51 * 5 * 32768 ~ 8M ops.
        # This is very fast in Python if optimized with bitmasks.
        
        return not is_2_colorable_fast(edges, n)

    def is_2_colorable_fast(edges, n):
        """
        Brute force check using bitmasks.
        Returns True if a valid 2-coloring exists.
        """
        if not edges:
            return True
            
        # Convert edges to bitmasks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Iterate all colorings
        # c is a bitmask of vertices colored 1. Vertices not in c are 0.
        for c in range(1 << n):
            valid = True
            for mask in edge_masks:
                # Edge is monochromatic if:
                # 1. All vertices are 0: (mask & c) == 0
                # 2. All vertices are 1: (mask & c) == mask
                if (mask & c) == 0 or (mask & c) == mask:
                    valid = False
                    break
            if valid:
                return True
        return False

    # Local Search Strategy
    best_edges = None
    best_count = float('inf')
    
    # Try to find a solution on n=15 first, as 51 edges is known for n=15.
    # If we can't find one quickly, try n=12, n=14, etc.
    
    for n in [15, 14, 13, 12, 11, 10, 9]:
        if time.time() - start_time > timeout:
            break
            
        # Generate all possible edges for this n
        all_edges = [tuple(e) for e in itertools.combinations(range(n), 5)]
        
        # Random restarts
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Start with a random set of edges, size ~ 55
            # If we start too small, it's likely colorable.
            # If we start too large, we have to remove many.
            # 55 is a good start for 51.
            current_edges = set(random.sample(all_edges, min(55, len(all_edges))))
            
            # Check if it's already non-2-colorable
            if is_not_2_colorable(list(current_edges), n):
                # Try to minimize
                improved = True
                while improved and time.time() - start_time < timeout:
                    improved = False
                    # Try removing random edges
                    edges_list = list(current_edges)
                    random.shuffle(edges_list)
                    for e in edges_list:
                        if time.time() - start_time > timeout:
                            break
                        if len(current_edges) <= best_count:
                            # No point in optimizing further if we already found smaller
                            break
                            
                        test_edges = current_edges - {e}
                        if is_not_2_colorable(list(test_edges), n):
                            current_edges = test_edges
                            improved = True
                            if len(current_edges) < best_count:
                                best_count = len(current_edges)
                                best_edges = list(current_edges)
                            break # Restart the shuffle loop
            else:
                # If colorable, add edges until non-colorable
                # This might take a long time.
                # Skip this restart if it takes too long?
                # We'll just add a few random edges and see.
                # Actually, if it's colorable, we can't minimize it.
                # We need to find a non-colorable one first.
                # Random 55-edge hypergraphs on 15 vertices are likely colorable.
                # We might need a smarter initialization.
                
                # Let's try adding edges greedily
                available = set(all_edges) - current_edges
                available_list = list(available)
                random.shuffle(available_list)
                
                # Add edges one by one
                for e in available_list:
                    if time.time() - start_time > timeout:
                        break
                    current_edges.add(e)
                    # Check every few additions to save time?
                    # Or just check if size > 60 to avoid huge graphs
                    if len(current_edges) > 60:
                        break
                    if is_not_2_colorable(list(current_edges), n):
                        # Found a non-colorable core, now minimize
                        improved = True
                        while improved and time.time() - start_time < timeout:
                            improved = False
                            edges_list = list(current_edges)
                            random.shuffle(edges_list)
                            for e_rem in edges_list:
                                if time.time() - start_time > timeout:
                                    break
                                if len(current_edges) <= best_count:
                                    break
                                test_edges = current_edges - {e_rem}
                                if is_not_2_colorable(list(test_edges), n):
                                    current_edges = test_edges
                                    improved = True
                                    if len(current_edges) < best_count:
                                        best_count = len(current_edges)
                                        best_edges = list(current_edges)
                                    break
                        break # Found one, stop adding for this restart

    if best_edges is None:
        # Fallback: Return a known valid solution if we have one, or a complete graph on 9 vertices
        # Complete graph on 9 vertices has 126 edges.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
    else:
        # Convert tuples to lists for output
        edges_list = [list(e) for e in best_edges]
        # Determine n from edges
        max_v = -1
        for e in edges_list:
            for v in e:
                if v > max_v:
                    max_v = v
        n = max_v + 1
        return {'n': n, 'edges': edges_list}

if __name__ == "__main__":
    res = construct()
    print(res['n'], len(res['edges']))
