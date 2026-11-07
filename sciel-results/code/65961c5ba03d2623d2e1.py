"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    # Fano plane on 7 vertices (0-6), vertices 7,8 are isolated
    # This hypergraph is 3-chromatic and has 7 edges.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [2, 4, 5],
        [1, 4, 6],
        [2, 3, 6]
    ]
    
    # Find a valid 3-coloring
    def find_coloring():
        # Backtracking search for 3-coloring
        n = 9
        colors = [-1] * n
        
        def is_valid(v, c):
            # Check if assigning color c to v creates a monochromatic edge
            for e in edges:
                if v in e:
                    # Check all other vertices in e
                    others = [u for u in e if u != v]
                    if all(colors[u] == c for u in others if colors[u] != -1):
                        # If all others are already c, and we set v to c, it's mono
                        # But we need to be careful: if some are -1, we don't know yet
                        # Actually, simpler: check if assigning c to v makes any edge mono
                        pass
            
            # Simpler check: just check all edges containing v
            for e in edges:
                if v in e:
                    vals = [colors[u] if colors[u] != -1 else None for u in e]
                    # If we set v to c, what are the values?
                    test_vals = []
                    for u in e:
                        if u == v:
                            test_vals.append(c)
                        else:
                            test_vals.append(colors[u])
                    # If all are assigned and equal, it's invalid
                    if all(x is not None for x in test_vals) and len(set(test_vals)) == 1:
                        return False
            return True

        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if is_valid(v, c):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False

        if backtrack(0):
            return colors
        return None

    coloring = find_coloring()
    
    # Fallback if backtracking fails (should not happen)
    if coloring is None:
        # Try random
        import random
        for _ in range(1000):
            c = [random.randint(0, 2) for _ in range(9)]
            valid = True
            for e in edges:
                if c[e[0]] == c[e[1]] == c[e[2]]:
                    valid = False
                    break
            if valid:
                coloring = c
                break
    
    # Ensure types are Python int
    edges_out = [[int(u) for u in e] for e in edges]
    colors_out = [int(c) for c in coloring]

    return {'edges': edges_out, 'colors': colors_out}
# EVOLVE-BLOCK-END
