"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Pappus Configuration edges (9 edges on 9 vertices)
    # Vertices 0..8
    edges = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 4, 6),
        (1, 3, 6),
        (1, 5, 7),
        (2, 4, 7),
        (2, 3, 8),
        (0, 5, 8)
    ]
    
    # Convert to lists as required
    edges = [list(e) for e in edges]
    
    # Find a 3-coloring
    # We need colors c[v] in {0, 1, 2} such that for all edges e, c[e[0]], c[e[1]], c[e[2]] are not all equal.
    # Since Pappus is 3-chromatic, a 3-coloring exists.
    # We can find one via backtracking.
    
    n = 9
    colors = [-1] * n
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    def is_valid_coloring(c):
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                return False
        return True
    
    def backtrack(idx):
        if idx == n:
            return True
        for col in range(3):
            colors[idx] = col
            # Check if current assignment violates any edge where all 3 vertices are assigned
            valid = True
            for e in edges:
                if colors[e[0]] != -1 and colors[e[1]] != -1 and colors[e[2]] != -1:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
            if valid:
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
        return False
    
    if backtrack(0):
        # Ensure types are int
        final_colors = [int(c) for c in colors]
        # Final check
        for e in edges:
            if final_colors[e[0]] == final_colors[e[1]] == final_colors[e[2]]:
                # Should not happen, but fallback
                pass
        
        return {'edges': edges, 'colors': final_colors}
    else:
        # Fallback: The seed solution (10 edges) if Pappus fails for some reason
        # But Pappus is known to be 3-chromatic, so this shouldn't happen.
        seed_edges = [list(e) for e in itertools.combinations(range(5), 3)]
        seed_colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
        # Verify seed colors are valid for seed edges
        # K5(3) with this coloring?
        # 0,0,1,1,2 on 0..4
        # Edge 0,1,2: 0,0,1 (ok)
        # Edge 0,1,3: 0,0,1 (ok)
        # Edge 0,1,4: 0,0,2 (ok)
        # Edge 0,2,3: 0,1,1 (ok)
        # Edge 0,2,4: 0,1,2 (ok)
        # Edge 0,3,4: 0,1,2 (ok)
        # Edge 1,2,3: 0,1,1 (ok)
        # Edge 1,2,4: 0,1,2 (ok)
        # Edge 1,3,4: 0,1,2 (ok)
        # Edge 2,3,4: 1,1,2 (ok)
        # This works.
        return {'edges': seed_edges, 'colors': seed_colors}

# EVOLVE-BLOCK-END
