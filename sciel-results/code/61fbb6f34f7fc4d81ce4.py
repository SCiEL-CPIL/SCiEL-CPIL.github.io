import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 150s kill
    
    n = 16
    # Build edge list and index mapping
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            edges.append((i,j))
            idx += 1
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    k4_list = []
    k4_edge_indices = []
    for quad in itertools.combinations(range(n), 4):
        edge_idxs = []
        for a in range(4):
            for b in range(a+1, 4):
                u, v = quad[a], quad[b]
                if u > v:
                    u, v = v, u
                edge_idxs.append(edge_idx[(u,v)])
        k4_list.append(quad)
        k4_edge_indices.append(edge_idxs)
    num_k4 = len(k4_list)  # 1820
    
    # For each edge, which K4s does it belong to?
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4i, eidxs in enumerate(k4_edge_indices):
        for ei in eidxs:
            edge_to_k4s[ei].append(k4i)
    
    # Fast validity check using NumPy
    # coloring is an array of shape (120,)
    # For each K4, we need at least 5 distinct colors
    
    def count_colors(coloring):
        return len(set(coloring.tolist()))
    
    def is_valid(coloring):
        # coloring is numpy array of int
        # For each K4, check distinct colors >= 5
        for eidxs in k4_edge_indices:
            cols = coloring[eidxs]
            if len(set(cols.tolist())) < 5:
                return False
        return True
    
    # Faster: precompute K4 edge index array
    k4_eidx_arr = np.array(k4_edge_indices, dtype=np.int32)  # shape (1820, 6)
    
    def is_valid_fast(coloring):
        # coloring: numpy array shape (120,)
        # k4_eidx_arr: shape (1820, 6)
        # For each K4, get the 6 colors and check distinct count
        # Use a trick: for each K4, sort the 6 colors and count unique
        # Or: check if any 2 pairs are equal in a way that reduces to <5
        # Actually, just check: number of distinct >= 5
        # This means: NOT (all 6 same) AND NOT (5 same + 1 diff) AND NOT (4 same + 2 same) AND NOT (4 same + 1 + 1) AND NOT (3 same + 3 same) AND NOT (3 same + 2 same + 1) AND NOT (3 same + 1 + 1 + 1) AND NOT (2 same + 2 same + 1 + 1) AND NOT (2 same + 2 same + 2 same)
        # Simpler: distinct < 5 means distinct <= 4
        # Let's just do it directly for speed
        
        # Get all K4 color arrays: shape (1820, 6)
        k4_colors = coloring[k4_eidx_arr]
        
        # For each K4, count distinct colors
        # A K4 has < 5 distinct colors iff it has <= 4 distinct colors
        # Check: for each K4, are there at most 4 distinct values?
        
        # Fast check: a set of 6 numbers has < 5 distinct values iff 
        # there exist two pairs that are equal (counting multiplicity)
        # More precisely: distinct < 5 means the 6 values span at most 4 distinct numbers
        
        # Let's use: sort each row and check gaps
        # Or: for each K4, check if any value appears 3+ times, or two values each appear 2+ times, etc.
        
        # Simplest reliable method:
        # distinct_count = 6 - (number of "redundant" elements)
        # But that's complex. Let's just use a vectorized approach.
        
        # For each K4 (row of 6 values), compute number of distinct values
        # We can do: for each row, count how many of the 6 positions have a value 
        # that first appears at that position.
        
        # Alternative: use the fact that distinct < 5 iff 
        # (max frequency >= 3) or (two values have frequency >= 2)
        
        # Let me just do it with a loop but optimized
        # Actually, for 1820 K4s, a Python loop might be too slow.
        
        # Let's use NumPy:
        # For each row, sort it
        sorted_k4 = np.sort(k4_colors, axis=1)
        # Count distinct: number of positions where sorted_k4[i] != sorted_k4[i-1], plus 1
        # diff = sorted_k4[:, 1:] - sorted_k4[:, :-1]
        # distinct = 1 + (diff != 0).sum(axis=1)
        
        diff = sorted_k4[:, 1:] - sorted_k4[:, :-1]
        distinct_counts = 1 + (diff != 0).sum(axis=1)
        
        return np.all(distinct_counts >= 5)
    
    # Start with a valid coloring: all edges distinct
    coloring = np.arange(num_edges, dtype=np.int32)
    
    # Greedy color merging
    # For each pair of colors (c1, c2), check if merging c2 into c1 keeps validity
    # Optimization: only consider pairs where both colors are present
    
    def get_color_edges(coloring):
        """Return dict: color -> list of edge indices"""
        color_to_edges = {}
        for i, c in enumerate(coloring):
            c = int(c)
            if c not in color_to_edges:
                color_to_edges[c] = []
            color_to_edges[c].append(i)
        return color_to_edges
    
    def try_merge(coloring, c1, c2):
        """Try to merge color c2 into c1. Return (success, new_coloring)"""
        # Create new coloring
        new_coloring = coloring.copy()
        mask = (new_coloring == c2)
        new_coloring[mask] = c1
        
        if is_valid_fast(new_coloring):
            return True, new_coloring
        return False, None
    
    # Main greedy loop
    best_coloring = coloring.copy()
    best_num_colors = num_edges
    
    while time.time() - start_time < TIME_LIMIT:
        color_to_edges = get_color_edges(best_coloring)
        colors = list(color_to_edges.keys())
        if len(colors) < 2:
            break
        
        merged = False
        # Sort colors by frequency (try merging less frequent colors first)
        colors_sorted = sorted(colors, key=lambda c: len(color_to_edges[c]))
        
        for i in range(len(colors_sorted)):
            if time.time() - start_time > TIME_LIMIT:
                break
            c1 = colors_sorted[i]
            for j in range(i+1, len(colors_sorted)):
                c2 = colors_sorted[j]
                success, new_coloring = try_merge(best_coloring, c1, c2)
                if success:
                    best_coloring = new_coloring
                    best_num_colors = len(set(best_coloring.tolist()))
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            # Try random perturbations
            # Pick a random edge and change its color to a random other color
            # Then try to merge more
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Random restart with different initial coloring
            # Try modular colorings
            for mod in [15, 16, 17, 19, 23]:
                if time.time() - start_time > TIME_LIMIT:
                    break
                # Color edge (i,j) with (i+j) % mod
                c = np.zeros(num_edges, dtype=np.int32)
                for k, (u, v) in enumerate(edges):
                    c[k] = (u + v) % mod
                if is_valid_fast(c):
                    # Use this as starting point
                    current = c.copy()
                    # Do greedy merging
                    for _ in range(100):
                        if time.time() - start_time > TIME_LIMIT:
                            break
                        cte = get_color_edges(current)
                        cs = list(cte.keys())
                        if len(cs) < 2:
                            break
                        cs_sorted = sorted(cs, key=lambda x: len(cte[x]))
                        m = False
                        for a in range(len(cs_sorted)):
                            for b in range(a+1, len(cs_sorted)):
                                s, nc = try_merge(current, cs_sorted[a], cs_sorted[b])
                                if s:
                                    current = nc
                                    m = True
                                    break
                            if m:
                                break
                        if not m:
                            break
                    nc = len(set(current.tolist()))
                    if nc < best_num_colors:
                        best_coloring = current
                        best_num_colors = nc
            
            # If still stuck, try random edge recoloring
            if time.time() - start_time > TIME_LIMIT:
                break
            
            # Perturb: change a few edges' colors
            current = best_coloring.copy()
            for _ in range(5):
                e = random.randint(0, num_edges-1)
                current[e] = random.randint(0, best_num_colors - 1)
            
            # Try to repair and merge
            # First, check if valid. If not, try to fix.
            # This is complex, so let's just try merging from this state
            # But it might be invalid. Let's check.
            if is_valid_fast(current):
                for _ in range(50):
                    if time.time() - start_time > TIME_LIMIT:
                        break
                    cte = get_color_edges(current)
                    cs = list(cte.keys())
                    if len(cs) < 2:
                        break
                    cs_sorted = sorted(cs, key=lambda x: len(cte[x]))
                    m = False
                    for a in range(len(cs_sorted)):
                        for b in range(a+1, len(cs_sorted)):
                            s, nc = try_merge(current, cs_sorted[a], cs_sorted[b])
                            if s:
                                current = nc
                                m = True
                                break
                        if m:
                            break
                    if not m:
                        break
                nc = len(set(current.tolist()))
                if nc < best_num_colors:
                    best_coloring = current
                    best_num_colors = nc
    
    return best_coloring.tolist()
