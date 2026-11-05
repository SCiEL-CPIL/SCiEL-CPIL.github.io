import itertools
import time
import random

def _precompute_c6_masks(n):
    """Precompute edge masks for all C6 cycles in K_n."""
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx += 1
    
    c6_masks = []
    # Generate all combinations of 6 vertices
    for vertices in itertools.combinations(range(n), 6):
        # Generate all unique C6 cycles on these 6 vertices
        # Fix v0 to avoid rotation duplicates, fix v1 to avoid reflection?
        # Simpler: iterate all permutations of the 6 vertices, check if it's a cycle, and normalize.
        # But 720 permutations is small.
        seen_cycles = set()
        for perm in itertools.permutations(vertices):
            # Check if perm is a valid cycle structure (v0-v1-...-v5-v0)
            # In K_n, any permutation is a valid cycle of edges.
            # But we want unique edge sets.
            # A cycle is defined by its set of 6 edges.
            edges = set()
            for k in range(6):
                u, v = perm[k], perm[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
            # Normalize the edge set
            edge_tuple = tuple(sorted(edges))
            if edge_tuple not in seen_cycles:
                seen_cycles.add(edge_tuple)
                mask = 0
                for e in edge_tuple:
                    mask |= (1 << edge_to_idx[e])
                c6_masks.append(mask)
    return c6_masks

def _construct_cyclic(n):
    """Try to find a cyclic coloring for K_n."""
    if n <= 1:
        return {'n': n, 'colors': []}
    
    # Differences are 1 to n//2
    max_diff = n // 2
    # For n=17, max_diff=8. Assignments: 4^8 = 65536
    # For n=15, max_diff=7. Assignments: 4^7 = 16384
    
    # Precompute C6 masks for this n
    # Note: This is expensive for n=17 (742k masks). 
    # But we only do it once.
    c6_masks = _precompute_c6_masks(n)
    
    # To speed up, we can filter c6_masks? No.
    
    start_time = time.time()
    timeout = 10.0
    
    for assign in itertools.product([0, 1, 2, 3], repeat=max_diff):
        if time.time() - start_time > timeout:
            return None
        
        # Build color masks
        masks = [0, 0, 0, 0]
        
        # Assign colors to edges
        # Edge (i, j) has diff d = (j - i) % n.
        # d is in 1..n-1.
        # We use min(d, n-d) as the canonical diff.
        # If canonical diff is k (1..max_diff), color is assign[k-1].
        
        # We need to build the full mask for each color.
        # Iterate all edges
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                if d > n - d:
                    d = n - d
                color = assign[d - 1]
                # Get edge index
                # We need a fast way to get index.
                # Precompute edge indices for all pairs
                # idx = i * n - i*(i+1)//2 + (j-i-1) ?
                # Let's just use a dict or a formula.
                # Formula for lexicographic index of (i,j) in n vertices:
                # idx = i*(2*n - i - 1)//2 + (j - i - 1)
                idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
                masks[color] |= (1 << idx)
        
        # Check validity
        valid = True
        for c in range(4):
            # If mask has fewer than 6 bits, no C6
            if masks[c].bit_count() < 6:
                continue
            # Check if any C6 is subset of masks[c]
            for c6_mask in c6_masks:
                if (c6_mask & ~masks[c]) == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Construct the result
            colors = [0] * (n * (n - 1) // 2)
            for i in range(n):
                for j in range(i + 1, n):
                    d = j - i
                    if d > n - d:
                        d = n - d
                    idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
                    colors[idx] = assign[d - 1]
            return {'n': n, 'colors': colors}
            
    return None

def construct():
    # Try n=17
    res = _construct_cyclic(17)
    if res:
        return res
        
    # Try n=15 (faster, likely to succeed)
    res = _construct_cyclic(15)
    if res:
        return res
        
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
