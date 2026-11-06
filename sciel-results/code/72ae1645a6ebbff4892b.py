"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            # Color based on the larger vertex index
            # This ensures no rainbow C4 as proven for linear orderings
            colors[idx] = j
            idx += 1
    return colors
# EVOLVE-BLOCK-END
