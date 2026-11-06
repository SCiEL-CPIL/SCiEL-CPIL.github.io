import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 9
    # Fallback: all edges for n=9 is 126, but we want min.
    # A known small non-2-colorable hypergraph?
    # The Fano plane is 3-uniform.
    # For 5-uniform, we rely on the search.
    # Initialize with a known valid one if possible, or just a large one.
    # The current code initializes with all edges of n=9 (126 edges).
    # We will try to improve this.
    
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    for n in range(9, 15):
        if time.time() > deadline - 15.0:
            break
            
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        # Precompute M0 and M1
        M0 = [0] * n
        M1 = [0] * n
        
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
                    
        all_edges = list(combinations(range(n), 5))
        E_count = len(all_edges)
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts
        # Smaller n allows more restarts
        if n <= 10:
            restarts = 100
        elif n <= 12:
            restarts = 50
        else:
            restarts = 20
            
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            covered = 0
            selected_indices = []
            remaining = set(range(E_count))
            
            # Greedy Set Cover
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                
                # Optimization: If we have many remaining, sampling might be faster?
                # But E is small (<2000). Full scan is fine.
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        
                if best_gain == 0:
                    break # Failed
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones:
                # Refinement Phase: Local Search to minimize size
                # We try to remove edges. If stuck, we try swaps.
                
                # 1. Deletion pass
                improved = True
                while improved:
                    improved = False
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        # Check redundancy
                        # Union of all other edges
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                                if new_covered == all_ones:
                                    break
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break # Restart deletion loop
                
                # 2. Swap pass (Local Search)
                # Try to swap an edge in C with an edge not in C to enable further deletion
                # We limit the number of swaps to avoid infinite loops or excessive time
                swap_attempts = 0
                max_swaps = 500 if n <= 10 else 100
                
                while swap_attempts < max_swaps and time.time() < deadline - 2.0:
                    swap_attempts += 1
                    swapped = False
                    # Pick a random edge from C to try to remove
                    if not selected_indices:
                        break
                    out_idx_pos = random.randint(0, len(selected_indices) - 1)
                    out_idx = selected_indices[out_idx_pos]
                    
                    # We want to replace out_idx with in_idx such that the cover is maintained
                    # AND ideally this new set allows an edge to be removed later.
                    # Simple heuristic: Just try to find ANY valid swap, then try to delete.
                    # To be more targeted, we can look for an in_idx that covers the "gap" left by out_idx
                    # Gap = covered & ~edge_masks[out_idx] ? No.
                    # The union of C\{out_idx} is U \ (parts only covered by out_idx).
                    # We need in_idx to cover the parts only covered by out_idx.
                    # Actually, we just need Union(C\{out_idx} U {in_idx}) == U.
                    
                    # Let's calculate the "need":
                    # need = all_ones & ~ (Union of C\{out_idx})
                    # This is expensive to calculate for every swap.
                    # Instead, just check validity.
                    
                    # To speed up, we can pre-calculate the union of C\{out_idx}?
                    # No, C changes.
                    
                    # Heuristic: Pick in_idx from remaining that has high overlap with edge_masks[out_idx]
                    # or high gain relative to the "hole".
                    
                    # Let's just pick a random candidate from remaining?
                    # Or pick the one with max intersection with edge_masks[out_idx]?
                    # Let's try max intersection with edge_masks[out_idx] to preserve coverage.
                    
                    best_in_idx = -1
                    best_overlap = -1
                    # Sample a few candidates from remaining if it's large
                    if len(remaining) > 50:
                        candidates = random.sample(list(remaining), 50)
                    else:
                        candidates = list(remaining)
                        
                    for in_idx in candidates:
                        # Check if swapping maintains cover
                        # Union of C\{out_idx} U {in_idx}
                        # This is O(|C|) bit ops.
                        # We can optimize: 
                        # Let U_others = Union of C\{out_idx}
                        # We need U_others | edge_masks[in_idx] == all_ones
                        # So edge_masks[in_idx] must cover (all_ones & ~U_others)
                        
                        # Calculate U_others
                        U_others = 0
                        for j in range(len(selected_indices)):
                            if j != out_idx_pos:
                                U_others |= edge_masks[selected_indices[j]]
                                if U_others == all_ones:
                                    break
                                    
                        if U_others == all_ones:
                            # out_idx is redundant, should have been caught in deletion
                            # But maybe deletion loop finished?
                            # Just remove it?
                            selected_indices.pop(out_idx_pos)
                            covered = U_others
                            swapped = True # Mark as change
                            break
                            
                        need = all_ones & ~U_others
                        # Check if in_idx covers need
                        if (edge_masks[in_idx] & need) == need:
                            # Valid swap
                            selected_indices[out_idx_pos] = in_idx
                            remaining.discard(in_idx)
                            remaining.add(out_idx)
                            covered = U_others | edge_masks[in_idx]
                            swapped = True
                            break
                            
                    if not swapped:
                        # No valid swap found in this iteration
                        # We can break or continue with different out_idx
                        # To avoid getting stuck, we break the swap loop for this restart
                        break
                        
                    # After a swap, try to delete again
                    improved = True
                    while improved:
                        improved = False
                        for i in range(len(selected_indices) - 1, -1, -1):
                            e_idx = selected_indices[i]
                            new_covered = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    new_covered |= edge_masks[selected_indices[j]]
                                    if new_covered == all_ones:
                                        break
                            if new_covered == all_ones:
                                selected_indices.pop(i)
                                covered = new_covered
                                improved = True
                                break
                                
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                
    return {'n': best_n, 'edges': best_edges}
