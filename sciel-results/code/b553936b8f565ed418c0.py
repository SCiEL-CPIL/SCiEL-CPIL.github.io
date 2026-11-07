import random
from itertools import combinations

def is_2_colorable(edges):
    # Try all 2^9 colorings? 512. Very fast.
    # If any coloring has no monochromatic edges, it is 2-colorable.
    # We want to find a hypergraph that is NOT 2-colorable.
    # So we check if there exists a 2-coloring with 0 mono edges.
    n = 9
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    # Try to find an 8-edge 3-chromatic hypergraph
    # Use a known candidate or random search
    # Known candidate:
    # Let's try a few random sets of 8 edges from the 84 possible
    # But random search might not find it quickly.
    # I will hardcode a known 8-edge 3-chromatic hypergraph.
    # After research, the following 8 edges are 3-chromatic:
    # (0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5), (3,5,7)
    # Wait, I need to be sure.
    # Let's use the 10-edge solution as a fallback.
    # To be safe and improve score, I will try to verify a specific 8-edge set.
    # If I can't be 100% sure, I'll stick to 10 or try 9.
    # Actually, I can implement a quick check in construct.
    
    # Hardcoded 8-edge candidate
    edges_8 = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
        (3, 5, 7)
    ]
    
    if not is_2_colorable(edges_8):
        # It is 3-chromatic (assuming it is 3-colorable, which it is)
        # Need to check 3-colorability? 
        # The task says "Minimize edge count among exactly three-chromatic".
        # If it's not 2-colorable, it's at least 3-chromatic.
        # We must ensure it IS 3-colorable.
        # Check if there is a 3-coloring with no mono edges.
        # 3^9 = 19683. Fast.
        def is_3_colorable(edges):
            n = 9
            for mask in range(3 ** n):
                valid = True
                for e in edges:
                    c0 = (mask // (3 ** e[0])) % 3
                    c1 = (mask // (3 ** e[1])) % 3
                    c2 = (mask // (3 ** e[2])) % 3
                    if c0 == c1 == c2:
                        valid = False
                        break
                if valid:
                    return True
            return False
        
        if is_3_colorable(edges_8):
            colors = [0, 1, 2, 0, 1, 2, 0, 1, 2] # Just some valid colors
            # Need to find the actual coloring? 
            # The return dict needs 'colors' which is a list of 9 labels.
            # The evaluator checks if the provided coloring is valid for the edges.
            # So I must provide a valid 3-coloring.
            # I can find it during the check.
            # Modify is_3_colorable to return the coloring.
            
            def find_3_coloring(edges):
                n = 9
                for mask in range(3 ** n):
                    valid = True
                    for e in edges:
                        c0 = (mask // (3 ** e[0])) % 3
                        c1 = (mask // (3 ** e[1])) % 3
                        c2 = (mask // (3 ** e[2])) % 3
                        if c0 == c1 == c2:
                            valid = False
                            break
                    if valid:
                        return [(mask // (3 ** i)) % 3 for i in range(n)]
                return None

            colors = find_3_coloring(edges_8)
            if colors:
                return {'edges': [list(e) for e in edges_8], 'colors': colors}
                
    # Fallback to 10 edges
    edges_10 = [list(e) for e in combinations(range(5), 3)]
    colors_10 = [0, 0, 1, 1, 2, 0, 0, 0, 0]
    return {'edges': edges_10, 'colors': colors_10}
