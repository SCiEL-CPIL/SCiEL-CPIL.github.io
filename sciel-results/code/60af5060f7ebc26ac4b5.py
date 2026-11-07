"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget to be safe
    
    # The Fano plane is a known 3-chromatic 3-uniform hypergraph with 7 edges.
    # We use it as a fallback solution.
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    best_edges = fano_edges
    best_colors = fano_colors
    
    # Fast 2-colorability check using bitmasks and backtracking
    def is_2_colorable_fast(edges):
        """
        Check if the hypergraph is 2-colorable.
        Returns True if 2-colorable, False otherwise.
        """
        n = 9
        # Precompute edge masks
        edge_masks = []
        for e in edges:
            mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
            edge_masks.append(mask)
        
        # Try all 2^9 = 512 possible 2-colorings
        for coloring in range(512):
            valid = True
            for mask in edge_masks:
                # Check if all three vertices have the same color
                # Get colors of the three vertices
                c0 = (coloring >> 0) & 1
                c1 = (coloring >> 1) & 1
                c2 = (coloring >> 2) & 1
                # Actually, we need to check the specific vertices in the edge
                # Let's extract the vertices from the mask
                v0 = -1
                v1 = -1
                v2 = -1
                m = mask
                count = 0
                while m:
                    if m & 1:
                        if count == 0:
                            v0 = 0
                        elif count == 1:
                            v1 = 1
                        elif count == 2:
                            v2 = 2
                        count += 1
                    m >>= 1
                # Get colors
                cv0 = (coloring >> v0) & 1
                cv1 = (coloring >> v1) & 1
                cv2 = (coloring >> v2) & 1
                if cv0 == cv1 == cv2:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # Even faster: just iterate all 512 colorings directly
    def is_2_colorable(edges):
        """
        Check if the hypergraph is 2-colorable by trying all 2^9 colorings.
        """
        for coloring in range(512):
            valid = True
            for e in edges:
                c0 = (coloring >> e[0]) & 1
                c1 = (coloring >> e[1]) & 1
                c2 = (coloring >> e[2]) & 1
                if c0 == c1 == c2:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # Find a 3-coloring for a given set of edges
    def find_3_coloring(edges):
        """
        Find a valid 3-coloring where no edge is monochromatic.
        Returns the coloring as a list of 9 ints, or None if not found.
        """
        n = 9
        colors = [-1] * n
        
        def bt(idx):
            if idx == n:
                for e in edges:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        return False
                return True
            
            v = idx
            for c in range(3):
                colors[v] = c
                # Check if any edge becomes monochromatic
                valid = True
                for e in edges:
                    if v in e:
                        c0 = colors[e[0]]
                        c1 = colors[e[1]]
                        c2 = colors[e[2]]
                        # Only check if all are assigned
                        if c0 != -1 and c1 != -1 and c2 != -1:
                            if c0 == c1 == c2:
                                valid = False
                                break
                if valid and bt(idx + 1):
                    return True
            colors[v] = -1
            return False
        
        if bt(0):
            return colors
        return None
    
    # Search for 6-edge 3-chromatic hypergraph
    all_triples = list(itertools.combinations(range(9), 3))
    random.seed(42)
    
    # Try random samples
    found = False
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        attempts += 1
        edges = random.sample(all_triples, 6)
        edges_list = [list(e) for e in edges]
        
        if not is_2_colorable(edges_list):
            # Found a 3-chromatic hypergraph!
            colors = find_3_coloring(edges_list)
            if colors:
                # Verify
                valid = True
                for e in edges_list:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
                if valid:
                    best_edges = edges_list
                    best_colors = colors
                    found = True
                    break
    
    # Ensure types are correct
    best_edges = [list(map(int, e)) for e in best_edges]
    best_colors = list(map(int, best_colors))
    
    return {'edges': best_edges, 'colors': best_colors}

# EVOLVE-BLOCK-END
