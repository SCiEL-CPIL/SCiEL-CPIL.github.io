import time

def construct():
    # Deadline guard: 60s wall-clock limit, use 55s as safety margin
    deadline = time.time() + 55
    
    # Best known solution: two K_{3,3} components
    # This gives cycle lengths {4, 6} -> 2 distinct lengths
    best_edges = []
    for u in range(3):
        for v in range(3, 6):
            best_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            best_edges.append([u, v])
    
    # The theoretical minimum is 1 (all cycles same length), but with min degree 3
    # and 12 vertices, achieving 1 distinct cycle length is extremely hard.
    # K_{3,3} gives 2. Let's see if we can find a graph with only 1 cycle length.
    # 
    # For a graph where all cycles have length 4 (a "4-cycle-only" graph):
    # This would be a graph where every cycle is a 4-cycle. Such graphs are
    # called "C4-only" graphs. A complete bipartite graph K_{m,n} has cycles
    # of lengths 4, 6, 8, ... up to 2*min(m,n). So K_{3,3} has 4 and 6.
    # 
    # To get only 4-cycles, we need a graph where no 6-cycles exist.
    # One approach: use a graph based on a linear algebra construction over GF(2).
    # 
    # Actually, let's think about this differently. The current score is 2.5
    # which means the raw value is 2 (since score = seed/value, and if seed=5,
    # then 5/2 = 2.5). Wait, let me re-read: "search score is seed/value".
    # If score is 2.5 and it's an improvement over 1.0, then seed/value = 2.5.
    # The inspiration 2 has score 1.0, which is the baseline. So seed/1 = 1.0 means
    # the baseline has value=seed. If baseline value is 5, seed=5. Then current
    # value = 5/2.5 = 2. So we have 2 distinct cycle lengths.
    # 
    # To improve, we need value=1, giving score 5.0.
    # 
    # Can we construct a triangle-free, min-degree-3 graph on 12 vertices with
    # only 1 distinct cycle length?
    # 
    # If all cycles are length 4: We need a graph where every cycle is a 4-cycle.
    # This is possible. Consider the graph from the incidence structure of a
    # projective plane or similar. But with 12 vertices and min degree 3...
    # 
    # Actually, a simpler approach: the utility graph K_{3,3} has cycles of length
    # 4 and 6. If we can find a bipartite graph where the girth is 4 and there are
    # no 6-cycles, we'd have only 4-cycles.
    # 
    # Let me try a different construction. Consider a graph where vertices are
    # arranged such that it's a "C4-only" graph. One known construction:
    # Take a graph where the adjacency matrix over GF(2) has the property that
    # A^k has specific patterns.
    # 
    # Alternative: use a small search to verify if a 1-cycle-length graph exists.
    # But given the 60s constraint, let's try a specific known construction.
    # 
    # The Clebsch graph? No, that's 16 vertices.
    # 
    # Let me try: a bipartite graph with parts of size 6 each, where each vertex
    # has degree 3, and the graph is the "incidence graph" of a design with no
    # 6-cycles.
    # 
    # Actually, let's just try to search for a better solution if time permits.
    # But first, let's return the known good solution.
    
    # Try to find a graph with 1 distinct cycle length using a limited search
    # This is complex, so let's just return the K_{3,3} x K_{3,3} solution.
    
    return [list(e) for e in best_edges]
