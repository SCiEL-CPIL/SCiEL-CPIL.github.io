"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    # Generate all weight-4 vectors with entries in {-1, 1}
    # Support: choose 4 positions out of 11, assign ±1 to each
    candidates = []
    supports = list(itertools.combinations(range(D), 4))
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(sup, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot product matrix
    # Two vectors are compatible if dot <= 2
    # For weight-4 ±1 vectors, ||v||^2 = 4, so ||v-w||^2 = 8 - 2*v·w
    # Need ||v-w|| >= 2 (max norm), so 8 - 2*v·w >= 4 => v·w <= 2
    
    # Precompute dot product matrix
    dot_matrix = candidates @ candidates.T
    
    # Build compatibility matrix: compatible[i,j] = (dot_matrix[i,j] <= 2) and i != j
    compatible = (dot_matrix <= 2)
    np.fill_diagonal(compatible, False)
    
    # Degree of each node (number of compatible neighbors)
    degrees = compatible.sum(axis=1)
    
    # Min-degree greedy initialization
    selected_set = set()
    available = set(range(n_cand))
    
    # Greedy: repeatedly pick node with minimum degree among available
    while available:
        if time.time() > deadline:
            break
        
        # Find node with minimum degree in the subgraph induced by available
        # For efficiency, compute degrees within available set
        avail_list = list(available)
        if not avail_list:
            break
        
        # Compute degrees within available
        min_deg = float('inf')
        min_node = -1
        for node in avail_list:
            deg = compatible[node, list(available)].sum()
            if deg < min_deg:
                min_deg = deg
                min_node = node
        
        if min_node == -1:
            break
        
        selected_set.add(min_node)
        available.discard(min_node)
        
        # Remove all nodes incompatible with min_node
        to_remove = []
        for node in list(available):
            if not compatible[node, min_node]:
                to_remove.append(node)
        for node in to_remove:
            available.discard(node)
    
    # Local search: 2-opt style swap (remove one, add one)
    best_selected = set(selected_set)
    
    def is_valid_set(selected):
        """Check if all pairs in selected are compatible."""
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                if not compatible[sel_list[i], sel_list[j]]:
                    return False
        return True
    
    # Try improving with swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(best_selected)
        
        # Try removing each node and adding a better one
        for remove_node in list(best_selected):
            if time.time() > deadline:
                break
            
            # Nodes that can replace remove_node
            # Must be compatible with all other selected nodes
            remaining = best_selected - {remove_node}
            
            for add_node in range(n_cand):
                if add_node in best_selected:
                    continue
                
                # Check if add_node is compatible with all remaining
                compatible_with_all = True
                for r in remaining:
                    if not compatible[add_node, r]:
                        compatible_with_all = False
                        break
                
                if compatible_with_all:
                    # This swap maintains validity
                    # Only do it if it leads to a larger set (or same size for diversity)
                    # Actually, we want to try to grow the set
                    new_set = best_selected - {remove_node} | {add_node}
                    # Try to extend new_set
                    new_available = set(range(n_cand)) - new_set
                    # Filter to only compatible nodes
                    new_available = {n for n in new_available if all(compatible[n, s] for s in new_set)}
                    
                    # Greedy extend
                    while new_available and time.time() < deadline:
                        # Pick min degree in new_available
                        min_deg = float('inf')
                        min_node = -1
                        for node in new_available:
                            deg = compatible[node, list(new_available)].sum()
                            if deg < min_deg:
                                min_deg = deg
                                min_node = node
                        
                        if min_node == -1:
                            break
                        
                        new_set.add(min_node)
                        new_available.discard(min_node)
                        
                        to_remove = [n for n in new_available if not compatible[n, min_node]]
                        for n in to_remove:
                            new_available.discard(n)
                    
                    if len(new_set) > current_size:
                        best_selected = new_set
                        current_size = len(new_set)
                        improved = True
                        break
            
            if improved:
                break
    
    # Convert back to vectors
    if len(best_selected) == 0:
        # Fallback: return seed solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates[list(best_selected)]
    return result
# EVOLVE-BLOCK-END
