"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    # The Fano plane on vertices 0-6 is 3-chromatic.
    # Adding isolated vertices 7 and 8 keeps it 3-chromatic on 9 vertices.
    # This uses 7 edges, which is the theoretical minimum for 3-chromatic 3-uniform hypergraphs.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # Vertices 7, 8 can be any color, e.g., 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify edges are non-monochromatic in this coloring
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # Fallback if manual coloring failed (should not happen)
            # Try a random search for a valid coloring
            import random
            for _ in range(1000):
                random_colors = [random.randint(0, 2) for _ in range(9)]
                valid = True
                for ee in edges:
                    if random_colors[ee[0]] == random_colors[ee[1]] == random_colors[ee[2]]:
                        valid = False
                        break
                if valid:
                    colors = random_colors
                    break
            else:
                # If still invalid, return a trivially valid but non-optimal structure?
                # No, the Fano plane MUST be 3-colorable.
                pass

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
