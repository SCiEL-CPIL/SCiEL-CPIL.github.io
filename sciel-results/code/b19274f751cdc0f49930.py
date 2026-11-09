"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Balanced complete bipartite graph K_{8,8} on vertices 0..15.
    # Diameter is exactly 2: any two vertices in the same part have
    # distance 2 (via any vertex in the other part), and vertices in
    # opposite parts are adjacent (distance 1).
    #
    # Edge-criticality: removing any edge (u, v) with u in U, v in V
    # forces dist(u, v) >= 3 because the graph remains bipartite and
    # u, v are in opposite parts, so no odd-length path exists; the
    # only remaining paths have length >= 3. Hence diameter > 2.
    #
    # 64 edges is the maximum possible for a bipartite graph on 16
    # vertices (balanced split 8+8). This is the densest edge-critical
    # diameter-2 graph in the bipartite family.

    u_part = [0, 1, 2, 3, 4, 5, 6, 7]
    v_part = [8, 9, 10, 11, 12, 13, 14, 15]

    result = []
    for a in u_part:
        for b in v_part:
            result.append([int(a), int(b)])

    return result

# EVOLVE-BLOCK-END
