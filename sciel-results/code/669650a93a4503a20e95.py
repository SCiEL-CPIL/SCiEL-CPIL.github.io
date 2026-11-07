"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_diameter_masks(adj_masks, n):
    """
    Calculate diameter using bitmask adjacency.
    adj_masks: list of n integers, where bit j is set if edge (i,j) exists.
    Returns diameter or inf if disconnected.
    """
    all_nodes = (1 << n) - 1
    max_dist = 0
    
    for start in range(n):
        visited = 1 << start
        frontier = 1 << start
        dist = 0
        
        while frontier:
            dist += 1
            next_frontier = 0
            f = frontier
            while f:
                lsb = f & -f
                idx = (lsb.bit_length() - 1)
                next_frontier |= adj_masks[idx]
                f -= lsb
            
            next_frontier &= ~visited
            visited |= next_frontier
            frontier = next_frontier
            
            if visited == all_nodes:
                break
        
        if visited != all_nodes:
            return float('inf')
        
        if dist > max_dist:
            max_dist = dist
            
    return max_dist

def _build_masks(edges, n):
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _is_valid_fast(edges, n):
    """
    Check if graph has diameter 2 and every edge is essential.
    """
    if not edges:
        return n == 1 # Trivial case, but n=16 here
    
    masks = _build_masks(edges, n)
    
    # 1. Check diameter is exactly 2
    if _bfs_diameter_masks(masks, n) != 2:
        return False
        
    # 2. Check each edge is essential
    for u, v in edges:
        # Create a copy of masks and remove edge (u,v)
        temp_masks = masks[:]
        temp_masks[u] &= ~(1 << v)
        temp_masks[v] &= ~(1 << u)
        
        new_diam = _bfs_diameter_masks(temp_masks, n)
        
        # Essential if new diameter > 2 OR disconnected (inf)
        if new_diam == 2:
            return False
            
    return True

def construct():
    n = 16
    deadline = time.time() + 85.0
    
    # Initialize with Star graph (15 edges)
    # This is a valid solution (diameter 2, all edges are bridges)
    best_edges = [[0, i] for i in range(1, n)]
    best_count = len(best_edges)
    
    current_edges = list(best_edges)
    current_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    random.seed(42)
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # 1. Try to add an edge
        candidates = [p for p in all_pairs if p not in current_set]
        if candidates:
            random.shuffle(candidates)
            added = False
            for u, v in candidates:
                if time.time() > deadline - 5.0:
                    break
                new_edges = current_edges + [[u, v]]
                if _is_valid_fast(new_edges, n):
                    current_edges = new_edges
                    current_set.add((u, v))
                    added = True
                    if len(current_edges) > best_count:
                        best_edges = list(current_edges)
                        best_count = len(best_edges)
                    no_improve_count = 0
                    break
            
            if not added:
                no_improve_count += 1
                
        # 2. If adding fails, try a swap (remove one, add one)
        if no_improve_count > 100:
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            removed_edge = current_edges.pop(idx)
            current_set.discard((min(removed_edge[0], removed_edge[1]), max(removed_edge[0], removed_edge[1])))
            
            candidates = [p for p in all_pairs if p not in current_set]
            random.shuffle(candidates)
            
            swapped = False
            for u, v in candidates:
                if time.time() > deadline - 5.0:
                    break
                new_edges = current_edges + [[u, v]]
                if _is_valid_fast(new_edges, n):
                    current_edges = new_edges
                    current_set.add((u, v))
                    swapped = True
                    if len(current_edges) > best_count:
                        best_edges = list(current_edges)
                        best_count = len(best_edges)
                    no_improve_count = 0
                    break
                    
            if not swapped:
                # Revert
                current_edges.append(removed_edge)
                current_set.add((min(removed_edge[0], removed_edge[1]), max(removed_edge[0], removed_edge[1])))
                no_improve_count += 100 # Penalize failed swap more

    # Ensure we return the best found
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
# EVOLVE-BLOCK-END
