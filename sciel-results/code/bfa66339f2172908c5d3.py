# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute base C6 masks on 6 labeled vertices 0..5
    # There are 15 distinct C6s
    base_c6_masks = []
    seen = set()
    # Generate all permutations of 0..5
    # To get unique masks, we can just generate all and dedupe
    # 720 permutations is small
    import itertools
    for perm in itertools.permutations(range(6)):
        mask = 0
        # Edge index function for 6 vertices
        # (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3, (0,5)->4
        # (1,2)->5, (1,3)->6, (1,4)->7, (1,5)->8
        # (2,3)->9, (2,4)->10, (2,5)->11
        # (3,4)->12, (3,5)->13
        # (4,5)->14
        def e_idx(i, j):
            if i > j: i, j = j, i
            # i < j
            # count edges (0,k) for k>i? No.
            # Standard formula: i*(5) - i*(i-1)//2 + (j - i - 1)
            # Let's verify:
            # i=0: 0 + j-1 = j-1. (0,1)->0, (0,5)->4. Correct.
            # i=1: 5 - 0 + j-2 = 3+j. (1,2)->5, (1,5)->8. Correct.
            # i=2: 10 - 1 + j-3 = 6+j. (2,3)->9, (2,5)->11. Correct.
            # i=3: 15 - 3 + j-4 = 8+j. (3,4)->12, (3,5)->13. Correct.
            # i=4: 20 - 6 + j-5 = 9+j. (4,5)->14. Correct.
            return i*5 - i*(i-1)//2 + (j - i - 1)
            
        for k in range(6):
            u = perm[k]
            v = perm[(k+1)%6]
            idx = e_idx(u, v)
            mask |= (1 << idx)
        if mask not in seen:
            seen.add(mask)
            base_c6_masks.append(mask)
            
    # base_c6_masks should have 15 elements
    
    def _precompute_c6_masks(n):
        """
        Precompute bitmasks for all C6s in K_n.
        Returns:
        - c6_masks: list of integers
        - edge_to_c6s: list of lists
        """
        num_edges = n*(n-1)//2
        
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i*(n-1) - i*(i-1)//2 + (j - i - 1)
        
        c6_masks = []
        # Map for each subset of 6 vertices
        # We iterate over all combinations of 6 vertices
        for subset in combinations(range(n), 6):
            # subset is a tuple of 6 sorted vertices
            # We map base vertex k to subset[k]
            # For each base mask, we create a new mask
            for bm in base_c6_masks:
                new_mask = 0
                # bm has bits for edges in 0..5
                # We need to map each edge (u,v) in bm to (subset[u], subset[v])
                # We can iterate through the 15 edges of K6
                for e in range(15):
                    if bm & (1 << e):
                        # Decode e to (u,v) in 0..5
                        # Inverse of e_idx for n=6
                        # e = i*5 - i*(i-1)//2 + (j - i - 1)
                        # We can just precompute a lookup table for n=6
                        pass # We'll use a lookup table for speed
                        
                # Let's use a lookup table for n=6 edge decoding
                # lookup6[e] = (u, v)
                for e in range(15):
                    if bm & (1 << e):
                        u, v = lookup6[e]
                        actual_u = subset[u]
                        actual_v = subset[v]
                        idx = edge_idx(actual_u, actual_v)
                        new_mask |= (1 << idx)
                        
                c6_masks.append(new_mask)
                
        # Build edge_to_c6s
        edge_to_c6s = [[] for _ in range(num_edges)]
        for i, mask in enumerate(c6_masks):
            # Iterate through all edges? No, just check bits.
            # But we need to know which edges are set.
            # We can iterate through the 15 base edges? No, the mask is on n edges.
            # We can just iterate through all edges in K_n and check bit.
            # num_edges is at most 190.
            # c6_masks has ~580k entries.
            # 580k * 190 is too slow.
            
            # Better: when constructing new_mask, we know which edges we set.
            # Let's modify the construction to also record edges.
            
        # Let's re-do precomputation to build edge_to_c6s efficiently.
        
        c6_masks = []
        edge_to_c6s = [[] for _ in range(num_edges)]
        
        for subset in combinations(range(n), 6):
            for bm in base_c6_masks:
                new_mask = 0
                # We need to map edges
                # Let's precompute the mapping for this subset
                # map_edge[e] = actual_idx
                # But we can just compute on the fly and add to edge_to_c6s
                
                idx_in_c6s = len(c6_masks)
                c6_masks.append(0) # Placeholder
                
                for e in range(15):
                    if bm & (1 << e):
                        u, v = lookup6[e]
                        actual_u = subset[u]
                        actual_v = subset[v]
                        idx = edge_idx(actual_u, actual_v)
                        new_mask |= (1 << idx)
                        edge_to_c6s[idx].append(idx_in_c6s)
                        
                c6_masks[idx_in_c6s] = new_mask
                
        return c6_masks, edge_to_c6s

    # Lookup table for n=6
    lookup6 = []
    for i in range(6):
        for j in range(i+1, 6):
            lookup6.append((i, j))
            
    # Note: lookup6 order must match e_idx order
    # e_idx(0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (0,5)=4
    # e_idx(1,2)=5, (1,3)=6, (1,4)=7, (1,5)=8
    # e_idx(2,3)=9, (2,4)=10, (2,5)=11
    # e_idx(3,4)=12, (3,5)=13
    # e_idx(4,5)=14
    # My loop:
    # i=0: j=1,2,3,4,5 -> (0,1), (0,2), (0,3), (0,4), (0,5) -> indices 0-4
    # i=1: j=2,3,4,5 -> (1,2), (1,3), (1,4), (1,5) -> indices 5-8
    # i=2: j=3,4,5 -> (2,3), (2,4), (2,5) -> indices 9-11
    # i=3: j=4,5 -> (3,4), (3,5) -> indices 12-13
    # i=4: j=5 -> (4,5) -> index 14
    # Matches perfectly.

    def _solve_with_deadline(n, dl):
        if time.time() > dl:
            return None
            
        num_edges = n*(n-1)//2
        c6_masks, edge_to_c6s = _precompute_c6_masks(n)
        
        color_masks = [0, 0, 0, 0]
        colors = [0] * num_edges
        
        def _randomize():
            nonlocal color_masks, colors
            color_masks = [0, 0, 0, 0]
            for e in range(num_edges):
                c = random.randint(0, 3)
                colors[e] = c
                color_masks[c] |= (1 << e)
                
        def _count_c6s():
            total = 0
            for c in range(4):
                cm = color_masks[c]
                for mask in c6_masks:
                    if (mask & cm) == mask:
                        total += 1
            return total
            
        _randomize()
        current_energy = _count_c6s()
        
        if current_energy == 0:
            return colors
            
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < dl:
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            delta = 0
            for idx in edge_to_c6s[e]:
                mask = c6_masks[idx]
                rest = mask ^ (1 << e)
                if (rest & color_masks[old_c]) == rest:
                    delta -= 1
                if (rest & color_masks[new_c]) == rest:
                    delta += 1
                    
            if delta <= 0 or random.random() < np.exp(-delta / T):
                colors[e] = new_c
                color_masks[old_c] &= ~(1 << e)
                color_masks[new_c] |= (1 << e)
                current_energy += delta
                if current_energy == 0:
                    return colors
            T *= cooling_rate
            
        return None

    # Try n from 20 down to 1
    for target_n in range(20, 0, -1):
        if time.time() > deadline - 5:
            break
            
        t_start = time.time()
        # Give more time to larger n? Or equal time?
        # Let's give 5s to each, but if we have more time, give it to the current attempt.
        # Actually, just run until deadline - 5s if it's the last one?
        # No, we want to try multiple n.
        # Let's allocate 10s per n.
        
        t_limit = min(deadline - 5, t_start + 10)
        
        res = _solve_with_deadline(target_n, t_limit)
        if res is not None:
            return {'n': target_n, 'colors': res}
            
    # Fallback
    n = 5
    colors = [0] * (n*(n-1)//2)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
