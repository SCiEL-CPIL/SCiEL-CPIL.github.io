import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for verts in k4s:
        edges_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edges_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edge_indices.append(edges_in_k4)
    
    # Convert to numpy array for fast operations
    k4_edges_np = np.array(k4_edge_indices, dtype=np.int32)
    num_k4s = len(k4s)
    
    def is_valid_np(colors_arr):
        """Fast vectorized validity check using NumPy."""
        # colors_arr is shape (120,)
        # For each K4, get the 6 edge colors
        # k4_edges_np is shape (num_k4s, 6)
        k4_colors = colors_arr[k4_edges_np]  # shape (num_k4s, 6)
        
        # For each K4, count distinct colors
        # A K4 is invalid if it has < 5 distinct colors
        # i.e., it has at most 4 distinct colors
        
        # Strategy: for each K4, check if any color appears >= 3 times 
        # (since 6 edges, if a color appears 3+ times, max distinct = 6-3+1 = 4)
        # Or more precisely: distinct < 5 means some color appears >= 3 times
        # (pigeonhole: 6 items into 4 bins means one bin has >= ceil(6/4)=2, but that's not enough)
        # Actually: 6 edges, < 5 distinct means at most 4 distinct
        # If max frequency is 2, then 6 edges / 2 per color = 3 colors max, so 4 distinct possible
        # If max freq is 3, then 3+1+1+1 = 6, so 4 distinct possible
        # If max freq is 4, then 4+1+1 = 6, so 3 distinct
        # So < 5 distinct iff max frequency >= 3
        
        # For each K4, compute max frequency of any color
        # This is tricky to vectorize directly. Let's use a different approach.
        
        # Alternative: for each K4, check all pairs of edges - if too many pairs share color
        # Simpler: just use np.unique per row but that's slow.
        
        # Better approach: A set of 6 colors has < 5 distinct iff there exists a color c 
        # that appears in at least 3 of the 6 positions.
        
        # For each K4 (row), check if any color appears >= 3 times.
        # We can do this by checking all triples of positions in each K4.
        
        # Positions: 0,1,2,3,4,5
        # Triples: C(6,3) = 20 triples
        # For each triple, check if all 3 positions have the same color
        
        triples = []
        for a in range(6):
            for b in range(a+1, 6):
                for c in range(b+1, 6):
                    triples.append((a, b, c))
        triples = np.array(triples, dtype=np.int32)  # shape (20, 3)
        
        # k4_colors[:, triples] would give shape (num_k4s, 20, 3)
        # Check if all 3 are equal for any triple
        # k4_colors[:, triples, 0] == k4_colors[:, triples, 1] == k4_colors[:, triples, 2]
        
        # This creates a boolean array of shape (num_k4s, 20)
        # If any entry is True, that K4 has a color appearing >= 3 times
        
        c0 = k4_colors[:, triples[:, 0]]  # (num_k4s, 20)
        c1 = k4_colors[:, triples[:, 1]]  # (num_k4s, 20)
        c2 = k4_colors[:, triples[:, 2]]  # (num_k4s, 20)
        
        has_triple_match = (c0 == c1) & (c1 == c2)  # (num_k4s, 20)
        invalid_k4s = np.any(has_triple_match, axis=1)  # (num_k4s,)
        
        return not np.any(invalid_k4s)
    
    # Initial coloring: (i + j) % 17
    colors = np.zeros(120, dtype=np.int32)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
    
    # Verify initial
    if not is_valid_np(colors):
        return colors.tolist()
    
    best_colors = colors.copy()
    best_count = len(set(colors.tolist()))
    
    # Merge loop with randomization to escape local minima
    no_improve_rounds = 0
    max_no_improve = 5
    
    while time.time() < deadline:
        current_colors_list = sorted(set(colors.tolist()))
        merged_any = False
        no_improve_this_round = False
        
        # Shuffle the order of merge attempts for diversity
        pairs_to_try = []
        for u in current_colors_list:
            for v in current_colors_list:
                if u != v:
                    pairs_to_try.append((u, v))
        
        random.shuffle(pairs_to_try)
        
        for u, v in pairs_to_try:
            if time.time() > deadline:
                break
            
            # Check if u is still present
            if u not in set(colors.tolist()):
                continue
            
            # Create a temporary copy
            temp_colors = colors.copy()
            temp_colors[temp_colors == u] = v
            
            # Check validity
            if is_valid_np(temp_colors):
                # Accept merge
                colors = temp_colors
                merged_any = True
                break
        
        if not merged_any:
            # Try random perturbation: pick a random edge and change its color
            # This can enable future merges
            if no_improve_rounds < max_no_improve:
                # Try several random perturbations
                found_better = False
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Start from best_colors or current?
                    # Let's try perturbing current state
                    perturbed = colors.copy()
                    
                    # Pick 1-3 random edges and assign random colors
                    num_changes = random.randint(1, 3)
                    for _ in range(num_changes):
                        edge_idx = random.randint(0, 119)
                        # Assign a random color from existing or a new one
                        existing_colors = list(set(perturbed.tolist()))
                        if random.random() < 0.7 and existing_colors:
                            perturbed[edge_idx] = random.choice(existing_colors)
                        else:
                            perturbed[edge_idx] = random.randint(0, 20)
                    
                    # Check if perturbed state is valid
                    if is_valid_np(perturbed):
                        # Try to merge from this perturbed state
                        perturbed_list = sorted(set(perturbed.tolist()))
                        for u in perturbed_list:
                            for v in perturbed_list:
                                if u == v:
                                    continue
                                temp = perturbed.copy()
                                temp[temp == u] = v
                                if is_valid_np(temp):
                                    colors = temp
                                    found_better = True
                                    break
                            if found_better:
                                break
                    
                    if found_better:
                        break
                
                if found_better:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
            else:
                # Too many failed rounds, try restarting from best
                colors = best_colors.copy()
                no_improve_rounds = 0
                
                # Try a different initial randomization
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    perturbed = best_colors.copy()
                    num_changes = random.randint(1, 5)
                    for _ in range(num_changes):
                        edge_idx = random.randint(0, 119)
                        perturbed[edge_idx] = random.randint(0, 25)
                    
                    if is_valid_np(perturbed):
                        colors = perturbed
                        break
        else:
            no_improve_rounds = 0
        
        # Update best
        current_count = len(set(colors.tolist()))
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
    
    # Final check
    if not is_valid_np(best_colors):
        # Fallback to original
        return [(i + j) % 17 for idx, (i, j) in enumerate(edges)]
    
    return best_colors.tolist()
