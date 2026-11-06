import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # We will try n from 9 to 14
    # For each n, we use a bitmask-based set cover approach to find a small hitting set
    # that covers all 2^n colorings.
    # An edge e "covers" a coloring c if e is monochromatic in c.
    # If the union of covered colorings is all 2^n, then no coloring is proper.
    
    for n in range(9, 15):
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
        
        # Precompute edge masks
        # mask[e] has bits set for colorings where e is monochromatic
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts
        # n=9: 512 colorings, C(9,5)=126 edges. Can do many restarts.
        # n=10: 1024 colorings, C(10,5)=252 edges.
        # n=11: 2048 colorings, C(11,5)=462 edges.
        # n=12: 4096 colorings, C(12,5)=792 edges.
        # n=13: 8192 colorings, C(13,5)=1287 edges.
        # n=14: 16384 colorings, C(14,5)=2002 edges.
        
        # Greedy Set Cover is O(N * E * bit_ops)
        # N is number of edges in solution (~50), E is total edges.
        # For n=12, E=792. 50 * 792 * (bit ops on 4096-bit int) is very fast.
        
        restarts = 50 if n <= 10 else (20 if n <= 12 else 5)
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Randomized Greedy
            # We can randomize the tie-breaking or the initial order.
            # Standard greedy: pick edge with max gain.
            # To add randomness, we can pick from top-k candidates or shuffle initial list.
            # Shuffling initial list doesn't help much for greedy gain.
            # Instead, we can use a "noisy" greedy: with prob p, pick random from top 10.
            # Or just standard greedy with different random seeds for tie-breaking?
            # Let's stick to standard greedy but maybe try different vertex orderings? No, edges are fixed.
            # Let's just run standard greedy. It is deterministic for a given set of masks.
            # To get different results, we need to perturb the process.
            # Idea: Randomly remove a few edges from the pool before starting?
            # Or: Randomly permute the edges list, and in case of ties, pick the first one.
            # Since gain is an integer, ties are common.
            
            covered = 0
            selected_indices = []
            remaining = list(range(len(all_edges)))
            random.shuffle(remaining) # This affects tie-breaking in the loop below if we iterate in order
            
            # Optimization: We can maintain a list of (gain, idx) and sort?
            # Or just scan. Scanning E items is fast.
            
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                
                # Iterate through remaining
                # To make it faster, we could break early if we find gain == total_uncovered?
                # total_uncovered = (all_ones ^ covered).bit_count()
                # If we find an edge that covers all remaining, we are done.
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        # If gain is 0, we are stuck (unless covered == all_ones which is handled by while)
                        # Actually if best_gain stays 0, we break.
                        
                if best_gain == 0:
                    break # Failed to cover all
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones:
                # We have a valid cover. Now refine by deleting redundant edges.
                # This is crucial for minimization.
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    # Try to remove each edge
                    # Random order to escape local minima of deletion?
                    # Deletion is deterministic if we check in order.
                    # Let's shuffle the selected_indices order to check?
                    # Or just check in reverse order of addition.
                    
                    # To be safe and thorough, check all.
                    # Since len(selected_indices) is small, this is fast.
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        # Check if e_idx is redundant
                        # Calculate OR of all other edges
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                                
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break # Restart loop to check again from end? Or continue?
                            # Restarting is safer to ensure no other edges became redundant due to this removal?
                            # Actually, removing an edge cannot make another edge redundant if it wasn't before?
                            # Wait. If A is redundant w.r.t S, and we remove B, A might still be redundant.
                            # If A was NOT redundant w.r.t S, removing B might make it redundant? No.
                            # If A is needed to cover bit x, and B doesn't cover x, removing B doesn't help A.
                            # If B covers x, and A covers x, and A is the only other one covering x...
                            # If we remove B, A becomes the ONLY one covering x, so A is NOT redundant.
                            # So removing an edge can only make other edges LESS redundant, not more?
                            # Wait.
                            # Let S = {e1, e2, e3}.
                            # e1 covers {1,2}, e2 covers {2,3}, e3 covers {1,3}.
                            # Union is {1,2,3}.
                            # Is e1 redundant? Union(e2,e3) = {1,2,3}. Yes.
                            # Remove e1. S={e2,e3}.
                            # Is e2 redundant? Union(e3) = {1,3} != {1,2,3}. No.
                            # Is e3 redundant? Union(e2) = {2,3} != {1,2,3}. No.
                            # So removal of e1 didn't make e2/e3 redundant.
                            
                            # Counter example:
                            # e1 covers {1}, e2 covers {1,2}, e3 covers {2}.
                            # Union {1,2}.
                            # e1 redundant? Union(e2,e3)={1,2}. Yes.
                            # Remove e1. S={e2,e3}.
                            # e2 redundant? Union(e3)={2} != {1,2}. No.
                            # e3 redundant? Union(e2)={1,2} != {1,2}? Wait. e2 covers {1,2}. So Union(e2)={1,2}.
                            # So e3 IS redundant in {e2,e3}.
                            # So removing e1 made e3 redundant!
                            # Thus, we MUST restart the deletion loop after a removal.
                            pass
                    
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                # print(f"Found better: n={n}, edges={best_count}")
                
    return {'n': best_n, 'edges': best_edges}
