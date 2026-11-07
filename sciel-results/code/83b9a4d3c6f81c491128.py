"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the edge list of K_{8,8}, the maximum-edge triangle-free graph
    on 16 vertices with diameter exactly 2 where every edge is critical.
    
    K_{8,8} has 64 edges. By Turán's theorem, no triangle-free graph on 16
    vertices can have more than floor(16^2/4) = 64 edges. Since any
    edge-critical diameter-2 graph must be triangle-free (a triangle would
    make one of its edges non-critical), 64 is the maximum.
    """
    n = 16
    half = n // 2
    edges = []
    # Partitions are 0..7 and 8..15
    for u in range(half):
        for v in range(half, n):
            # Ensure types are int
            edges.append([int(u), int(v)])
    return edges

# EVOLVE-BLOCK-END
