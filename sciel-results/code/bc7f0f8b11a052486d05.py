import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # We will try n from 9 to 13
    # The known minimum is 51 edges.
    # We aim to find a cover of size <= 51.
    
    for n in range(9, 14):
        if time.time() > deadline - 10.0:
            break
            
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        # Precompute M0 and M1
        # M0[v] has bits set for colorings where v=0
        # M1[v] has bits set for colorings where v=1
        M0 = [0] * n
        M1 = [0] * n
        
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
                    
        all_edges = list(combinations(range(n), 5))
        E = len(all_edges)
        
        # Precompute edge masks
        # mask[e] has bits set for colorings where e is monochromatic
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts depends on n and time remaining
        # For small n, we can afford more restarts.
        restarts = 100 if n <= 10 else (50 if n <= 11 else 20)
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Initial Greedy Set Cover
            covered = 0
            selected_indices = []
            remaining = list(range(E))
            random.shuffle(remaining)
            
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        
                if best_gain == 0:
                    break # Failed to cover all
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered != all_ones:
                continue
                
            # 2. Local Search: Remove and Repair
            current_S = selected_indices
            current_size = len(current_S)
            no_improve_count = 0
            
            # We try to reduce the size
            # Limit iterations per restart to avoid spending too much time on one
            max_ls_iter = 200 if n <= 10 else 100
            
            for ls_iter in range(max_ls_iter):
                if time.time() > deadline - 2.0:
                    break
                
                if not current_S:
                    break
                    
                # Pick a random edge to try removing
                # Bias towards edges that cover less unique area? 
                # For now, random.
                idx_to_remove = random.choice(current_S)
                
                # Remove it
                temp_S = [x for x in current_S if x != idx_to_remove]
                
                # Calculate uncovered
                covered = 0
                for idx in temp_S:
                    covered |= edge_masks[idx]
                
                uncovered = all_ones & ~covered
                
                if uncovered == 0:
                    # Edge was redundant
                    current_S = temp_S
                    current_size -= 1
                    no_improve_count = 0
                    continue
                
                # Repair: Add edges to cover 'uncovered'
                # We only consider edges NOT in temp_S
                # Optimization: Only consider edges that intersect 'uncovered'
                
                new_S = list(temp_S)
                # To speed up repair, we can pre-filter candidates
                # But E is small enough.
                
                # Create a set for fast lookup of current_S
                S_set = set(new_S)
                
                # Get candidates
                # We can just iterate all edges not in S_set
                # But we need to be careful with the loop.
                
                # Greedy repair
                while uncovered != 0:
                    best_gain = 0
                    best_idx = -1
                    
                    # Iterate through all edges not in new_S
                    # This is O(E) per step.
                    for idx in range(E):
                        if idx in S_set:
                            continue
                        gain = (edge_masks[idx] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                            
                    if best_gain == 0:
                        break # Stuck
                        
                    new_S.append(best_idx)
                    S_set.add(best_idx)
                    uncovered &= ~edge_masks[best_idx]
                    
                if uncovered == 0 and len(new_S) < current_size:
                    current_S = new_S
                    current_size = len(new_S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 20:
                        break # Restart local search for this greedy run? 
                        # Or just break out of LS and try next greedy restart.
                        # Breaking LS is safer to save time for new restarts.
                        
            if current_size < best_local_count:
                best_local_count = current_size
                best_local_edges = [all_edges[i] for i in current_S]
                
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                # print(f"Found better: n={n}, edges={best_count}") # Silent to save time? No, print is fast.
                
    # Fallback
    if best_edges is None or len(best_edges) > 200:
        # This shouldn't happen, but just in case
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
