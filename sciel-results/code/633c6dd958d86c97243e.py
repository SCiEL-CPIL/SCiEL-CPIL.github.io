"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _has_c6_bitmask(v, i, c, adj_mask, adj_list):
        """
        Check if adding edge (i,v) with color c creates a C6 in color c.
        Uses bitmask operations for speed.
        """
        # For each pair (d, a) where d < i, a < i, and adj[d][v] == c, adj[i][a] == c
        # we need to check if there exists b such that adj[a][b] == c and adj[b][d] == c
        # and all of {v, i, d, a, b} are distinct
        for d in range(i):
            if adj_mask[c][d] & (1 << v):
                continue
            # d is NOT connected to v in color c, so we can't form v-d edge
            # Actually we need d connected to v in color c
            pass
        
        # Better approach: iterate over all pairs (d, a) with d < a < i
        # where adj[d][v] == c and adj[i][a] == c
        # Then check if there's b with adj[a][b] == c and adj[b][d] == c
        for d in range(i):
            if not (adj_mask[c][d] >> v) & 1:
                continue
            for a in range(d+1, i):
                if not (adj_mask[c][i] >> a) & 1:
                    continue
                # Now check if there exists b (not in {v, i, d, a}) with adj[a][b]==c and adj[b][d]==c
                # adj_mask[c][a] has bits for all neighbors of a in color c
                # adj_mask[c][d] has bits for all neighbors of d in color c
                # We need b in adj_mask[c][a] & adj_mask[c][d] but not in {v, i, d, a}
                common = adj_mask[c][a] & adj_mask[c][d]
                # Remove v, i, d, a from common
                mask_remove = (1 << v) | (1 << i) | (1 << d) | (1 << a)
                if common & ~mask_remove:
                    return True
        return False

    def _constrained_backtrack(n_target, deadline):
        # adj_mask[c][v] is a bitmask of neighbors of v in color c
        adj_mask = [[0]*n_target for _ in range(4)]
        # adj_list[c][v] is a set of neighbors of v in color c (for iteration)
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        # adj[i][j] stores the color, -1 if unassigned
        adj = [[-1]*n_target for _ in range(n_target)]
        
        # Precompute edge order: for each vertex v, which edges to assign in what order
        # We'll assign edges in order: for v in 1..n-1, for i in 0..v-1
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            def _assign(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    return _search(v + 1)
                
                i = idx
                
                # Try colors in order of remaining capacity (fewest neighbors first)
                # This is a heuristic: try the most constrained color first
                color_order = sorted(range(4), key=lambda c: len(adj_list[c][i]))
                
                for c in color_order:
                    if _has_c6_bitmask(v, i, c, adj_mask, adj_list):
                        continue
                    
                    # Assign
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_mask[c][i] |= (1 << v)
                    adj_mask[c][v] |= (1 << i)
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    
                    if _assign(idx + 1):
                        return True
                    
                    # Unassign
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_mask[c][i] &= ~(1 << v)
                    adj_mask[c][v] &= ~(1 << i)
                    adj_list[c][i].discard(v)
                    adj_list[c][v].discard(i)
                
                return False
            
            return _assign(0)
        
        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj[i][j])
            return colors
        return None

    def _verify(colors, n):
        if n < 6:
            return True
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for C6 in each color using bitmask
        for c in range(4):
            # Build adjacency masks for color c
            mask = [0]*n
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] == c:
                        mask[i] |= (1 << j)
                        mask[j] |= (1 << i)
            
            # For each pair (i, j) with edge in color c, check if there's a path of length 4
            # from i to j through 3 intermediate vertices, all edges in color c
            for i in range(n):
                for j in range(i+1, n):
                    if not (mask[i] >> j) & 1:
                        continue
                    # Look for paths i-a-b-d-j where all edges are in color c
                    # a != b != d, all distinct from i, j
                    for a in range(n):
                        if a == i or a == j:
                            continue
                        if not (mask[i] >> a) & 1:
                            continue
                        for b in range(n):
                            if b == i or b == j or b == a:
                                continue
                            if not (mask[a] >> b) & 1:
                                continue
                            for d in range(n):
                                if d == i or d == j or d == a or d == b:
                                    continue
                                if not (mask[b] >> d) & 1:
                                    continue
                                if (mask[d] >> j) & 1:
                                    return False
        return True

    best_n = 1
    best_colors = []
    
    for n_target in range(2, 18):
        if time.time() > deadline - 10:
            break
        
        colors = _constrained_backtrack(n_target, deadline - 5)
        if colors is not None:
            if _verify(colors, n_target):
                best_n = n_target
                best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
