"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Generate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Build conflict graph: two edges conflict if they are opposite edges of a 2-face
    # Two parallel edges (same direction) conflict if their endpoints differ in exactly one other bit
    conflict = [set() for _ in range(m)]
    
    # Group edges by direction
    by_dir = [[] for _ in range(d)]
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        dir_bit = diff.bit_length() - 1
        by_dir[dir_bit].append(idx)
    
    # For each direction, check pairs of parallel edges
    for dir_bit in range(d):
        dir_edges = by_dir[dir_bit]
        for i in range(len(dir_edges)):
            for j in range(i + 1, len(dir_edges)):
                idx1 = dir_edges[i]
                idx2 = dir_edges[j]
                x1, y1 = edges[idx1]
                x2, y2 = edges[idx2]
                # Check if they form a 2-face: endpoints differ in exactly one bit (other than dir_bit)
                # x1 and x2 should differ in exactly one bit, and that bit is not dir_bit
                diff12 = x1 ^ x2
                # The bit that differs must not be dir_bit
                if diff12 & (1 << dir_bit) == 0 and diff12.bit_count() == 1:
                    # They conflict
                    conflict[idx1].add(idx2)
                    conflict[idx2].add(idx1)
    
    # MCS-inspired greedy: iteratively pick the edge with minimum degree among remaining
    remaining = set(range(m))
    selected = []
    
    while remaining:
        # Find edge with minimum degree in remaining
        min_deg = float('inf')
        best = -1
        for idx in remaining:
            deg = len(conflict[idx] & remaining)
            if deg < min_deg:
                min_deg = deg
                best = idx
        selected.append(best)
        remaining.remove(best)
        # Remove all conflicts of best from remaining
        for c in conflict[best]:
            if c in remaining:
                remaining.remove(c)
    
    # Convert back to edge list
    result = []
    for idx in selected:
        x, y = edges[idx]
        result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
