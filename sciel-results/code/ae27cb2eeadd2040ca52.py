import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute conflicts
    # For each edge, list the other 3 edges in each C4 it belongs to
    # If all 3 are present, adding this edge creates a C4
    conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            conflicts[idx].append((idx1, idx2, idx3))
    
    def is_valid_add(edge_idx, selected):
        for t in conflicts[edge_idx]:
            if all(x in selected for x in t):
                return False
        return True
    
    def greedy_fill(selected, available_indices):
        for idx in available_indices:
            if idx not in selected:
                if is_valid_add(idx, selected):
                    selected.add(idx)
        return selected
    
    start_time = time.time()
    time_limit = 110.0
    
    all_indices = list(range(n_edges))
    
    # Use multiple random seeds for diversity
    best_set = set()
    best_size = 0
    
    for seed in [42, 123, 456, 789, 1024, 2048, 333, 777, 999, 111]:
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        current_set = set()
        
        # Shuffle and greedy fill
        shuffled = list(all_indices)
        random.shuffle(shuffled)
        greedy_fill(current_set, shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Local search: try swapping edges
        # Represent as bitset for speed
        bitset = 0
        for idx in current_set:
            bitset |= (1 << idx)
        
        # Precompute for faster conflict checking
        # conflict_mask[e] = list of masks, where each mask has bits set for the 3 conflicting edges
        # Actually, let's just use the set-based approach but optimize the search
        
        iterations = 0
        max_iterations = 200000
        
        while iterations < max_iterations and time.time() - start_time < time_limit:
            iterations += 1
            
            # Try to add a random edge
            if len(current_set) < n_edges:
                # Find edges not in current_set
                # For efficiency, just pick random candidates
                for _ in range(50):  # Try up to 50 random adds
                    idx = random.randint(0, n_edges - 1)
                    if idx not in current_set:
                        if is_valid_add(idx, current_set):
                            current_set.add(idx)
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                            break
            
            # Try to swap: remove one edge, add another
            if current_set:
                remove_idx = random.choice(list(current_set))
                current_set.remove(remove_idx)
                
                # Try to add a different edge
                added = False
                for _ in range(50):
                    add_idx = random.randint(0, n_edges - 1)
                    if add_idx not in current_set and add_idx != remove_idx:
                        if is_valid_add(add_idx, current_set):
                            current_set.add(add_idx)
                            added = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                            break
                
                if not added:
                    # Put it back
                    current_set.add(remove_idx)
        
    # Final greedy fill on best_set
    remaining = [i for i in all_indices if i not in best_set]
    random.shuffle(remaining)
    greedy_fill(best_set, remaining)
    
    result = []
    for idx in best_set:
        result.append(list(edges[idx]))
        
    return result
