import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try to find a valid cyclic coloring for n=17
    result = _find_cyclic_coloring(17)
    if result:
        return result
        
    # If cyclic fails, try backtracking for smaller n
    # n=16 is 120 edges. n=15 is 105 edges.
    for n in [16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        budget = 10.0 if n >= 15 else 5.0
        end_time = time.time() + budget
        res = _backtracking_search(n, end_time)
        if res:
            return res
            
    # Fallback to seed
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

def _find_cyclic_coloring(n):
    # n must be prime for this specific simple check to be robust, but works for any n
    # We assume n=17
    if n != 17:
        return None
        
    # Pairs of differences: {1,16}, {2,15}, ..., {8,9}
    # We assign a color 0..3 to each pair index 1..8
    # There are 4^8 = 65536 possibilities
    
    # Precompute valid C6 checks for each color set
    # A color set S is a subset of {1..8} (representing pairs)
    # The actual differences are d and n-d for d in S
    
    def has_c6(color_set):
        # color_set is a frozenset of pair indices {1..8}
        # Differences available: d, n-d for d in color_set
        diffs = []
        for d in color_set:
            diffs.append(d)
            diffs.append(n - d)
            
        # Check all sequences of 6 differences from diffs
        # To speed up, we can use recursion or itertools.product
        # 4^6 = 4096 combinations per color class.
        # We have 4 color classes.
        # Total checks per assignment: 4 * 4096 = 16384.
        # 65536 assignments * 16384 = ~1 billion operations. Too slow in Python.
        
        # Optimization:
        # We only need to check if ANY color class has a C6.
        # We can precompute for each subset of pairs if it has a C6.
        # There are 2^8 = 256 subsets.
        # For each subset, check if it has a C6.
        # Then for an assignment, we just check if any of the 4 color classes (which are subsets) has a C6.
        
        pass

    # Precompute bad subsets
    bad_subsets = set()
    for mask in range(1, 1 << 8):
        subset = set()
        for i in range(8):
            if mask & (1 << i):
                subset.add(i + 1)
                
        # Check if this subset has a C6
        if _subset_has_c6(subset, n):
            bad_subsets.add(mask)
            
    # Now iterate through assignments
    # assignment is a tuple of 8 colors, c[0] for pair 1, ..., c[7] for pair 8
    # We can iterate mask by mask? No, we need to assign colors.
    # We can iterate 4^8.
    # But we can prune: if a color class (set of pairs assigned to that color) is in bad_subsets, it's invalid.
    
    # Let's use a recursive search with pruning
    pair_colors = [-1] * 8
    color_masks = [0, 0, 0, 0] # Bitmask of pairs assigned to color c
    
    def backtrack(pair_idx):
        if pair_idx == 8:
            # Check if any color mask is in bad_subsets
            for c in range(4):
                if color_masks[c] in bad_subsets:
                    return False
            return True
            
        for c in range(4):
            # Symmetry breaking:
            # Pair 1 (index 0) must be color 0
            if pair_idx == 0 and c != 0:
                continue
            # Pair 2 (index 1) must be color 0 or 1
            # If pair 1 is 0, pair 2 can be 0 or 1. If 1, it introduces color 1.
            if pair_idx == 1 and c > 1:
                continue
            # Generally, the first time a new color is introduced, it must be the next unused color?
            # This is complex. Let's just use the mask check.
            
            new_mask = color_masks[c] | (1 << pair_idx)
            
            # Pruning: if new_mask is in bad_subsets, skip
            if new_mask in bad_subsets:
                continue
                
            color_masks[c] = new_mask
            pair_colors[pair_idx] = c
            
            if backtrack(pair_idx + 1):
                return True
                
            color_masks[c] ^= (1 << pair_idx)
            pair_colors[pair_idx] = -1
            
        return False
        
    if backtrack(0):
        # Construct the colors list
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 0: continue # shouldn't happen
                if d > n // 2:
                    d = n - d
                # d is now 1..8
                c = pair_colors[d - 1]
                colors.append(c)
        return {'n': n, 'colors': colors}
        
    return None

def _subset_has_c6(subset, n):
    # subset is a set of pair indices {1..8}
    # diffs are d, n-d
    diffs = []
    for d in subset:
        diffs.append(d)
        diffs.append(n - d)
        
    if len(diffs) < 6:
        return False
        
    # Check all sequences of length 6 from diffs
    # Use recursion for speed/pruning?
    # Or just itertools.product
    # 4^6 is small.
    
    def check_seq(d1, d2, d3, d4, d5, d6):
        s1 = d1 % n
        s2 = (s1 + d2) % n
        s3 = (s2 + d3) % n
        s4 = (s3 + d4) % n
        s5 = (s4 + d5) % n
        s6 = (s5 + d6) % n
        if s6 != 0:
            return False
        # Check distinctness
        verts = [s1, s2, s3, s4, s5]
        if len(set(verts)) < 5:
            return False
        if 0 in verts:
            return False
        return True

    # To avoid 4^6 * 256 * 65536, we precompute once.
    # 256 subsets * 4096 seqs = 1M checks. Fast enough.
    
    # Optimization: We can just iterate all 6-tuples from diffs
    # But diffs can have duplicates? No, d and n-d are distinct for n=17.
    # And d are distinct in subset.
    # So diffs has 2*len(subset) unique elements.
    
    # We can use a helper to generate tuples
    # For simplicity and speed in precomputation, we'll just do a nested loop if len(diffs) is small,
    # or use a recursive generator.
    
    # Given the small size, a simple recursive check is fine.
    def rec(k, current_sum, visited):
        if k == 6:
            if current_sum % n != 0:
                return False
            # Check distinctness of partial sums
            # We need to store partial sums
            return True # This logic is flawed, need partial sums
            
    # Let's stick to the explicit check in _subset_has_c6 using itertools.product for clarity, 
    # as it's only called 256 times.
    import itertools
    for seq in itertools.product(diffs, repeat=6):
        s = 0
        verts = []
        valid = True
        for d in seq:
            s = (s + d) % n
            if s == 0:
                valid = False
                break
            if s in verts:
                valid = False
                break
            verts.append(s)
        if valid and s == 0:
            return True
    return False

def _backtracking_search(n, deadline):
    # Fallback search
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    def check_c6(u, v, c):
        nu = adj[c][u]
        nv = adj[c][v]
        x1_mask = nu
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            if x1 == v: continue
            x4_mask = nv
            while x4_mask:
                lsb4 = x4_mask & -x4_mask
                x4 = lsb4.bit_length() - 1
                x4_mask ^= lsb4
                if x4 == u or x4 == x1: continue
                forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                nx1 = adj[c][x1] & ~forbidden
                nx4 = adj[c][x4] & ~forbidden
                vx2 = nx1
                while vx2:
                    lsb2 = vx2 & -vx2
                    x2 = lsb2.bit_length() - 1
                    vx2 ^= lsb2
                    if adj[c][x2] & nx4:
                        return True
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == num_edges:
            return True
        u, v = edges[idx]
        c_vals = [0, 1, 2, 3]
        if idx == 0:
            c_vals = [0]
        else:
            random.shuffle(c_vals)
        for c in c_vals:
            if check_c6(u, v, c):
                continue
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            if backtrack(idx + 1):
                return True
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
        return False

    for _ in range(100):
        if time.time() > deadline:
            break
        # Reset state
        for i in range(num_edges): colors[i] = -1
        for c in range(4):
            for i in range(n): adj[c][i] = 0
        if backtrack(0):
            return {'n': n, 'colors': colors}
    return None
