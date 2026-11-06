import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def get_mono_mask(n, edge):
        """
        Returns a bitmask where bit i is 1 if coloring i makes 'edge' monochromatic.
        Coloring i is an integer from 0 to 2^n - 1.
        Bit v of i indicates if vertex v is color 1.
        Edge is mono if all vertices are 0 or all are 1.
        """
        mask = 0
        # Iterate all colorings
        for c in range(1 << n):
            # Check if edge is monochromatic under coloring c
            # Get the bits for the vertices in the edge
            # If all 0 or all 1, it's mono
            is_mono = True
            first_val = (c >> edge[0]) & 1
            for v in edge[1:]:
                if ((c >> v) & 1) != first_val:
                    is_mono = False
                    break
            if is_mono:
                mask |= (1 << c)
        return mask

    def is_non_2colorable_bitmask(n, edges, mono_masks_dict):
        """
        Check if the hypergraph defined by 'edges' is non-2-colorable
        using precomputed mono masks.
        """
        if not edges:
            return False # Empty hypergraph is 2-colorable
        
        # Start with all colorings alive
        alive = (1 << (1 << n)) - 1
        
        for e in edges:
            mono_m = mono_masks_dict[e]
            alive &= ~mono_m
            if alive == 0:
                return True # No valid coloring left
        
        return alive == 0

    def greedy_find_and_minimize(n, max_edges=200):
        """
        Find a non-2-colorable hypergraph on n vertices with minimal edges
        using greedy selection with bitmask optimization.
        """
        all_edges = list(combinations(range(n), 5))
        m_total = len(all_edges)
        
        if m_total < 10:
            return None

        # Pre-compute mono masks for all edges
        # This is the expensive part, but n is small (10-12).
        # 2^n colorings * n vertices check.
        mono_masks_dict = {}
        mono_masks_list = []
        for e in all_edges:
            m = get_mono_mask(n, e)
            mono_masks_dict[e] = m
            mono_masks_list.append(m)
            
        # Greedy selection
        alive = (1 << (1 << n)) - 1
        selected_edges = []
        selected_indices = []
        used = [False] * m_total
        
        while alive != 0:
            if time.time() > deadline - 5:
                return None
                
            best_idx = -1
            best_kills = 0
            
            # Find edge that kills most alive colorings
            # kills = popcount(alive & mono_mask[i])
            for i in range(m_total):
                if used[i]:
                    continue
                kills = (alive & mono_masks_list[i]).bit_count()
                if kills > best_kills:
                    best_kills = kills
                    best_idx = i
            
            if best_idx == -1 or best_kills == 0:
                # Should not happen if alive != 0 and there are edges
                break
                
            # Add edge
            selected_edges.append(all_edges[best_idx])
            selected_indices.append(best_idx)
            used[best_idx] = True
            
            # Update alive
            alive &= ~mono_masks_list[best_idx]
            
            if len(selected_edges) > max_edges:
                return None
                
        if alive != 0:
            return None
            
        # Minimization
        # Try to remove edges
        # We check if removing an edge makes it 2-colorable.
        # If it becomes 2-colorable, the edge was necessary.
        # If it remains non-2-colorable, the edge was redundant, remove it.
        
        # To speed up minimization, we can check in reverse order of addition
        # or random order. Reverse is often good for greedy.
        
        final_edges = list(selected_edges)
        
        # We need to be careful: removing an edge might not be independent.
        # We iterate through the current list.
        # For each edge, try removing it.
        # Check if the rest is non-2-colorable.
        
        # Optimization: If the set is small, just brute force check.
        # is_non_2colorable_bitmask is fast.
        
        changed = True
        while changed and len(final_edges) > 1:
            changed = False
            # Try removing each edge
            # To avoid bias, maybe shuffle? Or just iterate.
            # Let's iterate from end to start (last added are most likely redundant? Not necessarily).
            # Let's try a few passes.
            for i in range(len(final_edges) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                
                # Try removing final_edges[i]
                candidate = final_edges[:i] + final_edges[i+1:]
                if is_non_2colorable_bitmask(n, candidate, mono_masks_dict):
                    final_edges = candidate
                    changed = True
                    break # Restart loop with new list
        
        return final_edges

    best_edges = None
    best_m = float('inf')
    
    # Try n from 10 to 14
    # n=10: 252 edges.
    # n=11: 462 edges.
    # n=12: 792 edges.
    # n=13: 1287 edges.
    # n=14: 2002 edges.
    
    # The known bound 51 is likely achievable for small n.
    # Let's try n=10, 11, 12 first.
    
    for n in [10, 11, 12, 13]:
        if time.time() > deadline - 15:
            break
        
        # Run greedy
        edges = greedy_find_and_minimize(n, 100)
        
        if edges is not None:
            m = len(edges)
            if m < best_m:
                best_m = m
                best_edges = edges
                
            if m <= 51:
                # Found optimal or near-optimal
                # Try to improve with different random seeds? 
                # Greedy is deterministic here.
                # We can break if we hit 51, as it's the known best.
                # But maybe we can find < 51? Unlikely.
                # Let's break to save time for other n if needed, but 51 is the target.
                break

    if best_edges is None:
        # Fallback: Random search with bitmask check
        for n in [10, 11, 12]:
            if time.time() > deadline - 5:
                break
            
            all_edges = list(combinations(range(n), 5))
            mono_masks_dict = {}
            mono_masks_list = []
            for e in all_edges:
                m = get_mono_mask(n, e)
                mono_masks_dict[e] = m
                mono_masks_list.append(m)
            
            rng = random.Random(n)
            for trial in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Try random subset of size 51
                if len(all_edges) < 51:
                    continue
                
                indices = rng.sample(range(len(all_edges)), 51)
                subset = [all_edges[i] for i in indices]
                
                if is_non_2colorable_bitmask(n, subset, mono_masks_dict):
                    # Found one! Minimize it.
                    # Simple minimization
                    final_edges = list(subset)
                    changed = True
                    while changed and len(final_edges) > 1:
                        changed = False
                        for i in range(len(final_edges) - 1, -1, -1):
                            if time.time() > deadline - 1:
                                break
                            candidate = final_edges[:i] + final_edges[i+1:]
                            if is_non_2colorable_bitmask(n, candidate, mono_masks_dict):
                                final_edges = candidate
                                changed = True
                                break
                    best_edges = final_edges
                    best_m = len(final_edges)
                    break
            if best_edges:
                break

    if best_edges is None:
        # Absolute fallback: Complete hypergraph on 9 vertices (126 edges)
        # Or just return a known valid one if we had one hardcoded? 
        # Let's construct one quickly if time permits, else return empty? 
        # No, must return valid.
        # n=9 complete is 126 edges.
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        best_edges = edges
        best_m = len(edges)

    return {'n': n, 'edges': best_edges}

if __name__ == '__main__':
    # Simple test
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
