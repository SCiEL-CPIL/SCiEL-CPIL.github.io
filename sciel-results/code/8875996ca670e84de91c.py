import time

def construct():
    start_time = time.time()
    
    # Candidate: Two disjoint K_3,3
    # Vertices 0-5 form one K_3,3
    # Vertices 6-11 form another K_3,3
    edges = []
    
    # First K_3,3: A={0,1,2}, B={3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_3,3: A={6,7,8}, B={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Verification:
    # 1. Undirected simple edges on vertices 0..11: Yes.
    # 2. Every vertex degree >= 3: In K_3,3, every vertex has degree 3. Yes.
    # 3. Triangle-free: K_3,3 is bipartite, so no odd cycles, specifically no triangles. Yes.
    # 4. Contains a cycle: K_3,3 contains 4-cycles. Yes.
    # 5. Distinct cycle lengths:
    #    In K_3,3, simple cycles can only have even lengths.
    #    Since the bipartition sets have size 3, the maximum cycle length is 2*3=6.
    #    Possible lengths: 4, 6.
    #    Are both present?
    #    4-cycle: 0-3-1-4-0 (edges (0,3), (3,1) no wait, 3 is in B, 1 is in A. 3-1 is not an edge.
    #    Let's trace a 4-cycle in K_3,3:
    #    A={0,1,2}, B={3,4,5}
    #    Path: 0 -> 3 -> 1 -> 4 -> 0.
    #    Edges: (0,3) in K_3,3? Yes. (3,1) in K_3,3? Yes (1-3). (1,4) in K_3,3? Yes. (4,0) in K_3,3? Yes.
    #    So 4-cycles exist.
    #    6-cycle: 0 -> 3 -> 1 -> 5 -> 2 -> 4 -> 0.
    #    Edges: (0,3), (3,1), (1,5), (5,2), (2,4), (4,0). All exist in K_3,3.
    #    So 6-cycles exist.
    #    No other even lengths possible (2 is not a simple cycle, 8 > 6).
    #    So distinct lengths = {4, 6}. Count = 2.
    
    # Is it possible to have 1 distinct cycle length?
    # If min degree >= 3, the graph is not a tree.
    # If all cycles have the same length L.
    # If L=4: A graph where all cycles are 4-cycles.
    #   Such graphs are known to exist (e.g., certain cages or specific constructions).
    #   However, for n=12, min degree 3.
    #   Let's check if a "4-cycle only" graph with n=12, min deg 3 exists.
    #   The utility graph K_3,3 has 4 and 6 cycles.
    #   Consider the cube graph Q_3. n=8, min deg 3. Cycles are 4 and 6.
    #   Consider two disjoint K_3,3. n=12. Cycles 4, 6.
    #   What about a graph with only 4-cycles?
    #   If all cycles are length 4, the graph is a "4-cycle graph" in some sense.
    #   It is known that for sufficiently large n, such graphs exist.
    #   But for n=12?
    #   Let's try to construct one.
    #   If we take a single component, it must have min degree 3.
    #   Smallest such graph is K_4 (has triangles, so no).
    #   Next is K_{3,3} (has 6-cycles).
    #   What about the cube Q_3? (Has 6-cycles).
    #   What about the utility graph? (Has 6-cycles).
    #   Is there a triangle-free graph with min degree 3 and no 6-cycles?
    #   Yes, e.g., the graph obtained by taking a 4-cycle and adding chords? No, that creates triangles or other cycles.
    #   Actually, it is a known result that any triangle-free graph with minimum degree 3 contains a cycle of length at least 4.
    #   Does it necessarily contain a cycle of length > 4?
    #   Consider the graph C_4 with all vertices doubled? No.
    #   Let's look at the "Herschel graph" or similar.
    #   Actually, if we assume the answer is 1, we need a graph where all simple cycles have the same length.
    #   If the length is 4, the graph is a "4-cycle-free" in the sense of longer cycles? No, "4-cycle only".
    #   It turns out that for n=12, it is impossible to have min degree 3 and only 4-cycles.
    #   Why? 
    #   If a graph has min degree 3, it has at least 1.5 * 12 = 18 edges.
    #   K_3,3 has 9 edges. Two K_3,3 have 18 edges.
    #   Any graph with n=12, min deg 3 has at least 18 edges.
    #   If it has exactly 18 edges, it is a 3-regular graph.
    #   The only 3-regular triangle-free graphs on 12 vertices are:
    #   1. Two disjoint K_3,3.
    #   2. The cube Q_3 plus 4 vertices? No, Q_3 is 8 vertices.
    #   3. The utility graph is 6 vertices.
    #   Let's list 3-regular triangle-free graphs on 12 vertices.
    #   - Two K_3,3.
    #   - One connected graph?
    #     The "Crown" graph?
    #     If there is a connected 3-regular triangle-free graph on 12 vertices, does it have only 4-cycles?
    #     It is known that the girth of a 3-regular graph on 12 vertices can be 4.
    #     But can it have *only* 4-cycles?
    #     If a 3-regular graph has girth 4, it must contain 4-cycles.
    #     Does it contain 6-cycles?
    #     In K_3,3, yes.
    #     In the connected 3-regular triangle-free graph on 12 vertices (if it exists), does it have 6-cycles?
    #     Likely yes, due to expansion properties.
    #     
    #   Therefore, 2 is likely the minimum for n=12.
    #   The current solution achieves 2.
    
    # Check time
    if time.time() - start_time > 59:
        return edges
        
    return edges
