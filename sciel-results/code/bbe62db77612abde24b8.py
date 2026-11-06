"""
Program for dense 20-vertex 4-cycle rainbow edge-coloured graph.

Strategy:
- Use a 4-partite graph with parts of size 5.
- For each pair of parts, include a perfect matching (5 edges).
- Total edges: C(4,2) * 5 = 6 * 5 = 30.
- Coloring: assign a unique color to each edge in the range 0..19 (some colors unused).
- Any 4-cycle in a complete 4-partite graph visits 4 distinct parts.
- In our graph, a 4-cycle must use one edge from each of two adjacent part-pairs in a cycle.
- Since we only have matchings between each pair of parts, a 4-cycle uses 4 edges, each from a distinct part-pair.
- We assign colors such that edges from different part-pairs get colors from disjoint sets.
  Specifically:
    Parts: A, B, C, D
    Edges A-B: colors 0-4
    Edges B-C: colors 5-9
    Edges C-D: colors 10-14
    Edges D-A: colors 15-19
    Edges A-C: we need to handle these too. Wait, in a 4-partite graph, edges between A and C also exist.
    Let's reconsider: we have 6 pairs. Each pair has 5 edges. Total 30 edges.
    We have 20 colors. So some colors must be reused across different part-pairs.
    
    For a 4-cycle, it uses edges from 4 distinct part-pairs (since it visits 4 distinct parts).
    We need the 4 colors on the 4 edges to be distinct.
    
    If we assign colors 0-4 to A-B, 5-9 to B-C, 10-14 to C-D, 15-19 to D-A, and then for A-C and B-D, we must reuse colors.
    But if a 4-cycle uses edges from A-B, B-C, C-D, D-A, the colors are from 4 disjoint sets, so distinct.
    What if a 4-cycle uses edges from A-C, C-B, B-D, D-A? 
    Parts: A, C, B, D. Edges: A-C, C-B (same as B-C), B-D, D-A.
    Colors: A-C (reused), B-C (5-9), B-D (reused), D-A (15-19).
    If A-C and B-D reuse colors from sets 0-19, we might get conflicts.
    
    Better approach: 
    Only include edges between parts that are "adjacent" in a cycle of parts: A-B, B-C, C-D, D-A.
    Omit A-C and B-D.
    Then the graph is a 4-partite graph with only the "cycle" edges between parts.
    Edges: A-B (5), B-C (5), C-D (5), D-A (5). Total 20 edges.
    This is a bipartite graph? No, it's 4-partite.
    Does it have 4-cycles?
    A cycle of length 4: A-B-C-D-A.
    Edges: one from A-B, one from B-C, one from C-D, one from D-A.
    Since each inter-part edge set is a matching, a 4-cycle is formed by choosing one edge from each.
    Number of 4-cycles: 5^4 = 625.
    
    Coloring:
    A-B edges: colors 0-4
    B-C edges: colors 5-9
    C-D edges: colors 10-14
    D-A edges: colors 15-19
    Any 4-cycle uses one edge from each set, so 4 distinct colors.
    
    Total edges: 20.
    The seed has 19 edges. This is an improvement.
    
    Can we do better?
    If we add edges within a part, say in A, we might create 4-cycles that use 2 edges from A and 2 from other parts.
    But the problem states "simple graph", so no self-loops. Edges within A are allowed.
    However, adding an edge within A might create a 4-cycle that is not rainbow.
    
    Let's stick with the 20-edge construction for now. It's a solid improvement over 19.
"""

def construct():
    n = 20
    # Partition vertices into 4 parts of size 5
    parts = [[i for i in range(5*j, 5*(j+1))] for j in range(4)]
    A, B, C, D = parts
    
    edges = []
    
    # Define matchings between adjacent parts
    # A-B
    for i in range(5):
        u = A[i]
        v = B[i]
        edges.append([u, v, i])  # colors 0-4
    
    # B-C
    for i in range(5):
        u = B[i]
        v = C[i]
        edges.append([u, v, 5 + i])  # colors 5-9
    
    # C-D
    for i in range(5):
        u = C[i]
        v = D[i]
        edges.append([u, v, 10 + i])  # colors 10-14
    
    # D-A
    for i in range(5):
        u = D[i]
        v = A[i]
        edges.append([u, v, 15 + i])  # colors 15-19
    
    return edges
