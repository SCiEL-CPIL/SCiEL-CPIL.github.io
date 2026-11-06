import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    def get_all_colorings(n):
        return set(range(1, (1 << n) - 1))

    def is_mono(c, mask):
        return (c & mask) == 0 or (c & mask) == mask

    def check_non_2colorable(n, edges):
        """
        Returns True if the hypergraph defined by edges on n vertices is NOT 2-colorable.
        """
        if not edges:
            return False
            
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        alive = get_all_colorings(n)
        
        for em in edge_masks:
            new_alive = set()
            for c in alive:
                if not is_mono(c, em):
                    new_alive.add(c)
            alive = new_alive
            if not alive:
                return True
        return False

    def minimize_edges(n, edges):
        """
        Tries to remove edges from the list while maintaining non-2-colorability.
        Returns the minimized list.
        """
        current_edges = list(edges)
        # Shuffle to make removal order non-deterministic but consistent per run?
        # Or just iterate.
        # To be safe, iterate in reverse or random order.
        random.shuffle(current_edges)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(current_edges)):
                if time.time() > deadline - 10:
                    break
                # Try removing edge i
                candidate = current_edges[:i] + current_edges[i+1:]
                if check_non_2colorable(n, candidate):
                    current_edges = candidate
                    improved = True
                    # Restart loop as indices shifted
                    break
        return current_edges

    def greedy_find(n, max_edges_limit=200):
        """
        Greedily selects edges to eliminate all 2-colorings.
        Returns list of edge tuples or None.
        """
        all_edges = list(combinations(range(n), 5))
        if len(all_edges) < 5:
            return None
            
        all_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            all_masks.append(mask)
        
        alive = get_all_colorings(n)
        selected_indices = []
        selected_set = set()
        
        while alive:
            if time.time() > deadline - 5:
                return None
            
            best_idx = -1
            best_kills = 0
            
            # Iterate through all unselected edges to find the one that kills the most alive colorings
            for i in range(len(all_edges)):
                if i in selected_set:
                    continue
                
                em = all_masks[i]
                kills = 0
                # Count kills
                # Optimization: If kills already exceeds best_kills, we can't stop early 
                # unless we are sure it won't exceed? No, we need the exact count or at least 
                # a lower bound. Exact count is safer.
                # However, if best_kills is already very high, we might skip.
                # For now, exact count.
                for c in alive:
                    if is_mono(c, em):
                        kills += 1
                
                if kills > best_kills:
                    best_kills = kills
                    best_idx = i
            
            if best_idx == -1 or best_kills == 0:
                # No edge can kill any more colorings? 
                # This implies the remaining alive colorings are not mono for any remaining edge.
                # This means we cannot make it non-2-colorable with the remaining edges?
                # Actually, if best_kills == 0, it means for every remaining edge, 
                # there is no alive coloring that is mono for it.
                # This means all alive colorings are "valid" for all remaining edges.
                # So we can't kill them. We are stuck.
                return None
            
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            
            # Update alive
            em = all_masks[best_idx]
            new_alive = set()
            for c in alive:
                if not is_mono(c, em):
                    new_alive.add(c)
            alive = new_alive
            
            if len(selected_indices) > max_edges_limit:
                return None
        
        # Found a set
        edges_found = [all_edges[i] for i in selected_indices]
        
        # Minimize
        edges_min = minimize_edges(n, edges_found)
        
        return edges_min

    best_result = None
    best_m = float('inf')
    
    # Try n from 10 to 14
    # n=10 is fast, might yield ~60-70 edges.
    # n=12 might yield ~50 edges.
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline - 15:
            break
            
        result = greedy_find(n)
        
        if result is not None:
            m = len(result)
            if m < best_m:
                best_m = m
                best_result = (n, result)
                
            # If we found a very good result, we can stop?
            # The theoretical minimum is > 28. 51 is known.
            # If we find <= 51, it's excellent.
            if m <= 51:
                # We could try to improve further, but time is limited.
                # Let's break to save time for other n if we want, 
                # but usually larger n is harder. 
                # If n=10 gives 51, n=12 might give 50? 
                # Let's continue to n=12 if time permits.
                pass

    if best_result is not None:
        n, edges = best_result
        return {'n': n, 'edges': [list(e) for e in edges]}
    
    # Fallback: Return a known valid hypergraph if possible, or complete graph
    # Complete graph on 9 vertices is 126 edges.
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
