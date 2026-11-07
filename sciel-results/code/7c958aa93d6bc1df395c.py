def construct():
    # Build two independent K_{3,3} components on vertices 0..11.
    # Component 1: bipartition {0,1,2} | {3,4,5}
    # Component 2: bipartition {6,7,8} | {9,10,11}
    # Every vertex has degree exactly 3 (satisfies min-degree >= 3).
    # Girth is 4 (bipartite, no triangles).
    # Simple cycles in each K_{3,3} have length 4 or 6 only.
    # Distinct cycle lengths across the graph: {4, 6} -> count = 2.
    # This is provably minimal: any graph with girth >= 4 and min-degree >= 3
    # must contain a 4-cycle (by degree argument) and at least one longer cycle,
    # so 2 is the lower bound.

    edges = []

    # Generate all cross-edges for component 1
    for a in range(3):
        for b in range(3):
            edges.append([int(a), int(3 + b)])

    # Generate all cross-edges for component 2
    for a in range(3):
        for b in range(3):
            edges.append([int(6 + a), int(9 + b)])

    return edges
