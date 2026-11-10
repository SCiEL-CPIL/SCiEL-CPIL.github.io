import itertools
import time

def construct():
    n = 16
    # Base coloring with 16 colors
    base_colors = [(i + j) % 16 for i in range(n) for j in range(i + 1, n)]
    
    # Check validity function
    def is_valid(colors):
        # colors is a list of 120 ints
        # Map to edge (i,j)
        # For each K4, check distinct colors
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Get indices of edges in the flat list
                        # Index formula: i * n - (i+1)//2 + (j-i-1) ? 
                        # Simpler: precompute index map
                        pass
        
        # Precompute index map for speed
        idx = {}
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx[(i,j)] = k
                k += 1
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        e1 = idx[(a,b)]
                        e2 = idx[(a,c)]
                        e3 = idx[(a,d)]
                        e4 = idx[(b,c)]
                        e5 = idx[(b,d)]
                        e6 = idx[(c,d)]
                        cs = {colors[e1], colors[e2], colors[e3], colors[e4], colors[e5], colors[e6]}
                        if len(cs) < 5:
                            return False
        return True

    # Precompute indices
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            idx[(i,j)] = k
            k += 1
            
    edges = list(itertools.combinations(range(n), 2))
    k4s = list(itertools.combinations(range(n), 4))
    
    # Precompute edge indices for each K4
    k4_edges = []
    for quad in k4s:
        a,b,c,d = quad
        k4_edges.append([
            idx[(a,b)], idx[(a,c)], idx[(a,d)],
            idx[(b,c)], idx[(b,d)],
            idx[(c,d)]
        ])
        
    def check_validity_fast(colors):
        for e_idxs in k4_edges:
            cs = {colors[i] for i in e_idxs}
            if len(cs) < 5:
                return False
        return True

    # Try to merge colors from the base 16-coloring
    # Base uses 0..15
    best_colors = 16
    best_coloring = base_colors
    
    # Try merging 1 pair
    for c1 in range(16):
        for c2 in range(c1+1, 16):
            new_colors = []
            for c in base_colors:
                if c == c1:
                    new_colors.append(0)
                elif c == c2:
                    new_colors.append(1)
                else:
                    new_colors.append(c)
            # Renumber to 0..14
            # Actually, we just need to count distinct colors used
            # But the "color value" doesn't matter, only equivalence classes
            # So we can just check validity of the merged list
            if check_validity_fast(new_colors):
                # Count distinct
                used = set(new_colors)
                if len(used) < best_colors:
                    best_colors = len(used)
                    best_coloring = new_colors
                    
    # Try merging 2 pairs (from the best 15-coloring or base?)
    # It's safer to iterate from base and try all 2-pair merges
    # But that's 120 * 120 / 2 ... too many?
    # \binom{16}{2} = 120. \binom{120}{2} is huge.
    # Better: Greedy merge.
    
    # Let's just return the best found so far (likely 15)
    # To get 14, we can do a second pass on the best 15-coloring
    if best_colors == 15:
        # Try merging two colors in the 15-coloring
        used_colors = sorted(set(best_coloring))
        # Map to 0..14
        mapping = {c: i for i, c in enumerate(used_colors)}
        compacted = [mapping[c] for c in best_coloring]
        
        for c1 in range(15):
            for c2 in range(c1+1, 15):
                new_colors = []
                for c in compacted:
                    if c == c1:
                        new_colors.append(0)
                    elif c == c2:
                        new_colors.append(1)
                    else:
                        new_colors.append(c)
                if check_validity_fast(new_colors):
                    used = set(new_colors)
                    if len(used) < best_colors:
                        best_colors = len(used)
                        best_coloring = new_colors
                        break
            if best_colors < 15:
                break

    # Return the coloring
    # Need to ensure colors are in 0..119? No, "120 integer colours in 0..119"
    # The problem says "120 integer colours ... in 0..119".
    # This implies the OUTPUT values should be in 0..119.
    # But we minimize the NUMBER of distinct colors used.
    # So if we use 15 colors, we can map them to 0..14.
    
    used = set(best_coloring)
    final_map = {c: i for i, c in enumerate(sorted(used))}
    final = [final_map[c] for c in best_coloring]
    
    return final

