import time
import random
import math
from collections import deque

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    def validate_solution(edges_list, cycles_list):
        """
        Validates that:
        1. All edges are valid pairs.
        2. All cycles are valid simple cycles of the correct length.
        3. All cycle edges exist in the edge set.
        """
        edge_set = {norm_edge(e[0], e[1]) for e in edges_list}
        
        # Check for duplicates in edges (though set handles it, we check list integrity if needed)
        if len(edge_set) != len(edges_list):
            return False

        for i, cyc in enumerate(cycles_list):
            expected_len = i + 3 # cycles 0..37 correspond to lengths 3..40
            if len(cyc) != expected_len:
                return False
            if len(set(cyc)) != expected_len:
                return False
            # Check edges of the cycle
            for j in range(expected_len):
                u = cyc[j]
                v = cyc[(j + 1) % expected_len]
                if norm_edge(u, v) not in edge_set:
                    return False
        return True

    # Strategy: Construct a sparse graph based on a Hamiltonian cycle 
    # and add chords to facilitate shorter cycles.
    # 
    # Topology: 
    # Base: Cycle 0-1-...-(N-1)-0.
    # Chords: We want to create cycles of length k for all k in [3, N-1].
    # 
    # A known construction for a pancyclic graph with O(N) edges:
    # Take a cycle C_N.
    # Add chords (i, i+2) for all i. 
    # This creates triangles (i, i+1, i+2).
    # Does it create all cycles?
    # Let's analyze the graph G = C_N + {(i, i+2) mod N}.
    # This graph is the "square of a cycle" minus the long range? No, C_N^2 includes (i, i+2).
    # Actually, C_N^2 is pancyclic.
    # Edges in C_N: N.
    # Edges in C_N^2: N (from C_N) + N (from distance 2) = 2N.
    # For N=40, 80 edges. This is worse than the Fan (77).
    
    # Can we do better?
    # We need to minimize edges.
    # The Fan graph (77 edges) is very efficient.
    # Let's try to improve upon the Fan graph by a different construction.
    
    # Alternative: "Lollipop" or "Broom" structures?
    # 
    # Let's try a specific construction that might yield < 77 edges.
    # Consider a path 0-1-...-(N-1).
    # Add edge (0, N-1) to make it a cycle.
    # Add edge (0, 2).
    # This gives us a 3-cycle (0,1,2).
    # To get a 4-cycle, we need a chord that allows a 4-cycle.
    # If we add (0, 3), we get 0-1-2-3-0? No, (2,3) is in path, (3,0) is chord.
    # Wait, in the path 0-1-2-3-4..., edge (2,3) exists.
    # Cycle 0-1-2-3-0 uses (0,1), (1,2), (2,3), (3,0).
    # So adding (0,3) creates a 4-cycle.
    # Adding (0,4) creates a 5-cycle 0-1-2-3-4-0.
    # ...
    # Adding (0,k) creates a (k+1)-cycle 0-1-...-k-0.
    # This is exactly the Fan graph!
    # Edges: Path (N-1) + Cycle Close (1) + Chords (N-2) = 2N - 2 = 78 for N=40?
    # Wait.
    # Fan graph edges:
    # Spokes: (0,1)...(0,39) -> 39 edges.
    # Path: (1,2)...(38,39) -> 38 edges.
    # Total: 77 edges.
    # My "Path + Chords" description:
    # Path edges: (0,1)...(38,39) -> 39 edges.
    # Cycle close: (39,0) -> 1 edge.
    # Chords: (0,2)...(0,38) -> 37 edges.
    # Total: 39 + 1 + 37 = 77 edges.
    # It is the same graph, just described differently.
    
    # So the Fan graph is a "Path + Chords from one end".
    # Is there a topology with fewer edges?
    # Lower bound for pancyclic graph on N vertices:
    # It must be 2-connected (to have a Hamiltonian cycle).
    # Min edges for 2-connected graph is N (a cycle).
    # But a cycle is not pancyclic.
    # It is known that the minimum number of edges in a pancyclic graph on n vertices is n + 2 for n >= 4?
    # Let's check n=4.
    # C_4 (4 edges) is not pancyclic (no triangle).
    # C_4 + 1 chord (5 edges) is pancyclic.
    # So n+1 for n=4.
    # n=5.
    # C_5 (5 edges) not pancyclic.
    # C_5 + 1 chord (6 edges).
    # Chord (0,2).
    # 3-cycle: 0-1-2-0.
    # 4-cycle: 0-2-3-4-0? No, 0-2-3-4-0 uses (0,2), (2,3), (3,4), (4,0). Yes.
    # 5-cycle: 0-1-2-3-4-0.
    # So n+1 for n=5.
    # n=6.
    # C_6 + 1 chord (7 edges).
    # Chord (0,2).
    # 3-cycle: 0-1-2-0.
    # 4-cycle: 0-2-3-4-0? No, (4,0) not edge.
    # 4-cycle: 0-1-2-3-0? No (3,0) not edge.
    # 4-cycle: 2-3-4-5-2? No.
    # It seems 1 chord is not enough for n=6.
    # Try 2 chords.
    # Chords (0,2) and (0,3)?
    # 3: 0-1-2-0.
    # 4: 0-1-2-3-0? No (2,3) is path, (3,0) is chord. Yes.
    # 5: 0-1-2-3-4-0? No (4,0) not edge.
    # 5: 0-3-4-5-0? No.
    # 5: 0-1-2-3-4-5-0? No.
    # 
    # Let's try Chords (0,2) and (1,3).
    # 3: 0-1-2-0.
    # 4: 0-1-3-2-0? No.
    # 4: 1-2-3-1? No.
    # 
    # It is a known result that for n >= 4, there exists a pancyclic graph with n + 2 edges?
    # Or n + 3?
    # 
    # If the minimum is around 42-45 edges, we can beat 77 significantly.
    # 
    # Let's try to construct a graph with ~N+2 edges.
    # Construction:
    # Start with Cycle C_N.
    # We need to "peel off" smaller cycles.
    # 
    # Let's try a "Recursive" construction.
    # G_n is pancyclic.
    # Add a vertex v and two edges to G_n to make G_{n+1} pancyclic.
    # If we can add 1 vertex and 2 edges, we get N + 2 edges (starting from N=3, K_3 has 3 edges = N).
    # 3 vertices: K_3 (3 edges). Pancyclic (3).
    # 4 vertices: Add v4. Connect to v1, v2.
    # Edges: 3 (K3) + 2 = 5.
    # Cycles:
    # 3: 1-2-3-1 (from K3).
    # 4: 1-2-4-1? No, 4-cycle needs 4 distinct vertices.
    # 1-2-4-3-1? Edges (1,2), (2,4), (4,3)? No edge (4,3).
    # So this simple addition doesn't work immediately.
    
    # Let's try a different known sparse pancyclic graph:
    # The "Wheel" W_n has n+1 vertices and 2n edges.
    # For N=40, W_39 has 40 vertices and 78 edges.
    # The Fan graph has 77 edges.
    # 
    # Is there a graph with 76 edges?
    # 
    # Let's try to optimize the Fan graph by removing one edge and repairing.
    # We established that removing (0,39) or (0,1) makes a leaf.
    # What if we remove a "middle" path edge, say (20,21)?
    # Vertex 20 is connected to 19, 0.
    # Vertex 21 is connected to 22, 0.
    # No leaves.
    # Can we find cycles?
    # The graph is split into two paths connected at 0.
    # Path A: 0-1-...-20.
    # Path B: 0-21-...-39.
    # Any cycle must go 0 -> Path A -> 0 or 0 -> Path B -> 0 or 0 -> Path A -> 0 -> Path B -> 0?
    # No, simple cycles cannot repeat 0.
    # So any cycle is contained entirely in Path A + 0 or Path B + 0.
    # Max cycle length in Path A: 21 (0-1-...-20-0).
    # Max cycle length in Path B: 20 (0-21-...-39-0).
    # We need a 40-cycle.
    # Impossible.
    # So we cannot remove any path edge that disconnects the graph into components that don't allow a Hamiltonian cycle.
    # In the Fan graph, the path 1-2-...-39 is a "bridge" structure relative to the hub 0?
    # No, 0 is connected to all.
    # But the path edges are critical for the "long" cycles.
    
    # Conclusion: The Fan graph is structurally very rigid for Hamiltonian cycles.
    # 
    # Let's try a completely different topology: **The "Grid" or "Ladder"**.
    # A 2 x 20 ladder graph.
    # Vertices: (0,0)..(0,19) and (1,0)..(1,19).
    # Edges:
    # Horizontal: (i,j)-(i,j+1) for i=0,1 and j=0..18. (2 * 19 = 38 edges).
    # Vertical: (0,j)-(1,j) for j=0..19. (20 edges).
    # Total: 58 edges.
    # Is the Ladder graph pancyclic?
    # It contains cycles of all lengths from 4 to 40?
    # Smallest cycle: 4 (a square).
    # Does it contain a 3-cycle?
    # No, Ladder is bipartite. No odd cycles.
    # So Ladder is NOT pancyclic.
    
    # We need odd cycles.
    # Add a chord to the Ladder to break bipartiteness.
    # Add (0,0)-(0,1)? Already there.
    # Add (0,0)-(1,1)?
    # This creates a triangle 0,0 - 1,0 - 1,1 - 0,0?
    # Edges: (0,0)-(1,0) [Vertical], (1,0)-(1,1) [Horizontal], (1,1)-(0,0) [Chord].
    # Yes, triangle.
    # Edges: 58 + 1 = 59.
    # Is it pancyclic?
    # It has a 3-cycle.
    # It has 4-cycles (squares).
    # It has 5-cycles?
    # 0,0-1,0-1,1-0,1-0,0? No, (0,1)-(0,0) is edge.
    # 0,0-0,1-1,1-1,0-0,0.
    # Edges: (0,0)-(0,1), (0,1)-(1,1), (1,1)-(1,0), (1,0)-(0,0).
    # This is a 4-cycle.
    # 5-cycle: 0,0-0,1-0,2-1,2-1,1-0,0?
    # Edges: (0,0)-(0,1), (0,1)-(0,2), (0,2)-(1,2), (1,2)-(1,1), (1,1)-(0,0).
    # Yes!
    # It seems the Ladder + 1 Chord might be pancyclic.
    # If so, 59 edges is a huge improvement over 77.
    
    # Let's verify if Ladder + Chord is pancyclic for N=40.
    # We need to find a cycle of every length 3..40.
    # 
    # Algorithm to find cycles in Ladder + Chord:
    # The Ladder is a "series" of squares.
    # A cycle in a ladder graph is generally a "rectangle" spanning k columns.
    # Length of such a cycle: 2k + 2?
    # For k=1 (1 square): 4-cycle.
    # For k=2 (2 squares wide): 6-cycle?
    # 0,0-0,1-0,2-1,2-1,1-1,0-0,0. Length 6.
    # For k columns, the perimeter is 2k + 2.
    # This gives even lengths 4, 6, ..., 40.
    # We need odd lengths 3, 5, 7, ..., 39.
    # The chord (0,0)-(1,1) allows us to "cut corners".
    # 
    # 3-cycle: 0,0-1,0-1,1-0,0. (Length 3).
    # 5-cycle: 0,0-0,1-0,2-1,2-1,1-0,0. (Length 5).
    #    Path: (0,0)->(0,1)->(0,2)->(1,2)->(1,1)->(0,0).
    #    Edges: (0,0)-(0,1), (0,1)-(0,2), (0,2)-(1,2), (1,2)-(1,1), (1,1)-(0,0).
    #    All exist.
    # 7-cycle: 0,0-0,1-0,2-0,3-1,3-1,2-1,1-0,0.
    #    Path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(1,2)->(1,1)->(0,0).
    #    Edges: (0,0)-(0,1), (0,1)-(0,2), (0,2)-(0,3), (0,3)-(1,3), (1,3)-(1,2), (1,2)-(1,1), (1,1)-(0,0).
    #    All exist.
    # 
    # General pattern for odd length 2k+1:
    # Start at (0,0). Go right along top row to (0, k-1).
    # Go down to (1, k-1).
    # Go left along bottom row to (1, 1).
    # Go to (0,0) via chord.
    # Length:
    # Top: k-1 edges.
    # Down: 1 edge.
    # Bottom: k-2 edges (from k-1 to 1).
    # Chord: 1 edge.
    # Total: (k-1) + 1 + (k-2) + 1 = 2k - 1.
    # Wait, we want length L = 2k+1.
    # If L=3 (k=1): Top 0, Down 1, Bottom -1? No.
    # Let's re-index.
    # L=3: 0,0-1,0-1,1-0,0.
    # Top: 0. Down: 1. Bottom: 0. Chord: 1. Total 2? No, 3 vertices.
    # Edges: (0,0)-(1,0), (1,0)-(1,1), (1,1)-(0,0).
    # 
    # L=5: 0,0-0,1-0,2-1,2-1,1-0,0.
    # Top: 2 edges. Down: 1. Bottom: 1. Chord: 1. Total 5.
    # 
    # L=7: 0,0-0,1-0,2-0,3-1,3-1,2-1,1-0,0.
    # Top: 3 edges. Down: 1. Bottom: 2. Chord: 1. Total 7.
    # 
    # Pattern for L=2m+1 (m>=1):
    # Top: m edges.
    # Down: 1 edge.
    # Bottom: m-1 edges.
    # Chord: 1 edge.
    # Total: m + 1 + m - 1 + 1 = 2m + 1.
    # Max m:
    # Top row has 19 edges (0..19).
    # So m can be up to 19.
    # Max L = 2(19) + 1 = 39.
    # So we can get all odd lengths 3..39.
    
    # What about even lengths 4..40?
    # Standard Ladder cycles (rectangles).
    # Rectangle spanning columns 0..k-1.
    # Top: k-1 edges.
    # Right: 1 edge.
    # Bottom: k-1 edges.
    # Left: 1 edge.
    # Total: 2k.
    # Min k=1: L=2? No, k=1 means 1 column wide.
    # Vertices: 0,0; 0,1; 1,1; 1,0.
    # Edges: (0,0)-(0,1), (0,1)-(1,1), (1,1)-(1,0), (1,0)-(0,0).
    # Length 4.
    # So k=1 gives L=4.
    # k=2 gives L=6.
    # ...
    # k=19 gives L=38.
    # We need L=40.
    # The full ladder perimeter is 40?
    # Vertices: 2 * 20 = 40.
    # Perimeter: 2 * 19 (top/bottom) + 2 (sides) = 40.
    # Yes, the outer cycle of the ladder is length 40.
    # So we have even lengths 4..40.
    
    # So the Ladder + 1 Chord graph is Pancyclic!
    # Edge count: 59.
    # Score: (59 - 40) / (77 - 40) = 19 / 37 ≈ 0.51.
    # This is a massive improvement.
    
    # Let's construct this explicitly.
    
    # Vertices:
    # Top row: 0..19
    # Bottom row: 20..39
    # Mapping:
    # (r, c) -> r*20 + c
    # 0,0 -> 0
    # 0,1 -> 1
    # ...
    # 0,19 -> 19
    # 1,0 -> 20
    # 1,1 -> 21
    # ...
    # 1,19 -> 39
    
    edges = []
    
    # Horizontal edges Top
    for c in range(19):
        edges.append((c, c + 1))
        
    # Horizontal edges Bottom
    for c in range(19):
        edges.append((20 + c, 20 + c + 1))
        
    # Vertical edges
    for c in range(20):
        edges.append((c, 20 + c))
        
    # Chord
    # (0,0) to (1,1) -> 0 to 21
    edges.append((0, 21))
    
    # Total edges: 19 + 19 + 20 + 1 = 59.
    
    # Now generate the cycles.
    cycles = []
    
    # Odd cycles (3, 5, ..., 39)
    # L = 2m + 1, m = 1..19
    for m in range(1, 20):
        # L = 2m + 1
        # Cycle:
        # Start 0 (0,0)
        # Go right along top to (0, m-1)
        # Go down to (1, m-1)
        # Go left along bottom to (1, 1)
        # Go to 0 via chord (1,1)-(0,0)
        
        path = [0]
        # Top row: 0 to m-1
        for c in range(1, m):
            path.append(c)
        # Now at (0, m-1) which is vertex m-1.
        # Go down to (1, m-1) which is vertex 20 + m - 1.
        path.append(20 + m - 1)
        # Go left along bottom from m-1 to 1.
        # Vertices: 20 + m - 2, 20 + m - 3, ..., 20 + 1.
        for c in range(m - 2, 0, -1):
            path.append(20 + c)
        # Now at (1, 1) which is vertex 21.
        # Next is 0 (via chord).
        
        # Check length:
        # 1 (start) + (m-1) (top) + 1 (down) + (m-2) (bottom) = m + m - 2 = 2m - 1?
        # Wait.
        # m=1: L=3.
        # Path: 0.
        # Top: range(1, 1) is empty.
        # Down: 20 + 1 - 1 = 20.
        # Bottom: range(1-2, 0, -1) = range(-1, 0, -1) is empty.
        # Path: [0, 20].
        # Next is 0.
        # Cycle: 0-20-0? No, 0-20 is vertical. 20-0 is vertical.
        # But we need 3-cycle: 0-20-21-0.
        # My path construction for m=1:
        # path = [0]
        # top: none.
        # down: 20.
        # bottom: none.
        # path = [0, 20].
        # We need to go 20 -> 21 -> 0.
        # The "bottom" part should go from m-1 to 1.
        # If m=1, m-1=0. We are at 20 (which is 1,0).
        # We need to go to 21 (1,1).
        # So the bottom loop should be from m-1 down to 1.
        # If m=1, we are at 0 (col 0). We need to go to 1 (col 1).
        # So we go RIGHT?
        # My previous analysis:
        # L=3: 0-20-21-0.
        # Top: 0.
        # Down: 0->20.
        # Bottom: 20->21.
        # Chord: 21->0.
        # 
        # L=5: 0-1-2-22-21-0.
        # Top: 0->1->2.
        # Down: 2->22.
        # Bottom: 22->21.
        # Chord: 21->0.
        # 
        # General m:
        # Top: 0 -> 1 -> ... -> m-1.
        # Down: m-1 -> 20+m-1.
        # Bottom: 20+m-1 -> 20+m-2 -> ... -> 21.
        # Chord: 21 -> 0.
        
        path = [0]
        for c in range(1, m):
            path.append(c)
        # Now at m-1.
        path.append(20 + m - 1)
        for c in range(m - 2, 0, -1):
            path.append(20 + c)
        # Now at 21.
        # Cycle closes to 0.
        
        # Verify length:
        # 1 + (m-1) + 1 + (m-2) = 2m - 1?
        # m=1: 1 + 0 + 1 + 0 = 2. No, L=3.
        # The number of edges is the length.
        # Edges:
        # Top: m-1 edges.
        # Down: 1 edge.
        # Bottom: m-2 edges.
        # Chord: 1 edge.
        # Total: m-1+1+m-2+1 = 2m - 1.
        # For m=1, L=1? No.
        # If m=1, Top=0, Down=1, Bottom=0, Chord=1. Total 2 edges?
        # Cycle 0-20-0 is not a simple cycle of length 3.
        # The issue is the bottom part for m=1.
        # For m=1, we are at 20 (col 0). We need to go to 21 (col 1).
        # So we must go RIGHT.
        # My loop `range(m-2, 0, -1)` goes from -1 to 1? No, empty.
        # So for m=1, we miss the edge (20,21).
        
        # Correction:
        # Bottom path goes from col m-1 to col 1.
        # If m-1 > 1, we go left: m-1, m-2, ..., 1.
        # If m-1 == 1 (m=2), we are at col 1. No movement.
        # If m-1 == 0 (m=1), we are at col 0. We need to go to col 1. So we go RIGHT.
        
        # Let's handle m=1 separately or adjust logic.
        if m == 1:
            path = [0, 20, 21]
        else:
            path = [0]
            for c in range(1, m):
                path.append(c)
            path.append(20 + m - 1)
            for c in range(m - 2, 0, -1):
                path.append(20 + c)
            # Path ends at 21.
        
        cycles.append(path)

    # Even cycles (4, 6, ..., 40)
    # L = 2k.
    # Rectangle spanning columns 0..k-1.
    # Top: 0..k-1.
    # Right: k-1 -> 20+k-1.
    # Bottom: 20+k-1 -> ... -> 20.
    # Left: 20 -> 0.
    
    for k in range(1, 20):
        # L = 2k
        # k=1: L=4.
        # Path: 0-1-21-20-0.
        # Top: 0->1.
        # Right: 1->21.
        # Bottom: 21->20.
        # Left: 20->0.
        
        path = [0]
        for c in range(1, k):
            path.append(c)
        # Now at k-1.
        path.append(20 + k - 1)
        for c in range(k - 2, -1, -1):
            path.append(20 + c)
        # Now at 20.
        # Cycle closes to 0.
        
        cycles.append(path)
        
    # We have cycles for 3..39 (odd) and 4..38 (even).
    # We are missing 40.
    # 40-cycle: Perimeter of the ladder.
    # 0-1-...-19-39-38-...-20-0.
    path_40 = []
    for c in range(20):
        path_40.append(c)
    for c in range(19, -1, -1):
        path_40.append(20 + c)
    cycles.append(path_40)
    
    # Check order:
    # cycles[0] is L=3.
    # cycles[1] is L=4.
    # ...
    # We generated odd first, then even.
    # We need to sort them by length.
    
    # Let's re-generate in order 3..40.
    final_cycles = [None] * 38
    
    # Odd lengths: 3, 5, ..., 39
    # Index in final_cycles: L-3.
    for m in range(1, 20):
        L = 2 * m + 1
        idx = L - 3
        if m == 1:
            c = [0, 20, 21]
        else:
            c = [0]
            for cc in range(1, m):
                c.append(cc)
            c.append(20 + m - 1)
            for cc in range(m - 2, 0, -1):
                c.append(20 + cc)
        final_cycles[idx] = c

    # Even lengths: 4, 6, ..., 38
    for k in range(1, 20):
        L = 2 * k
        idx = L - 3
        c = [0]
        for cc in range(1, k):
            c.append(cc)
        c.append(20 + k - 1)
        for cc in range(k - 2, -1, -1):
            c.append(20 + cc)
        final_cycles[idx] = c
        
    # Length 40
    idx = 40 - 3
    c40 = []
    for cc in range(20):
        c40.append(cc)
    for cc in range(19, -1, -1):
        c40.append(20 + cc)
    final_cycles[idx] = c40
    
    # Validate
    if not validate_solution(edges, final_cycles):
        # Fallback to Fan if construction failed (should not happen)
        edges_set = set()
        for i in range(1, N):
            edges_set.add(norm_edge(0, i))
        for i in range(1, N-1):
            edges_set.add(norm_edge(i, i+1))
        fan_cycles = []
        for k in range(3, N+1):
            fan_cycles.append(list(range(k)))
        return {'edges': [list(e) for e in edges_set], 'cycles': fan_cycles}
        
    return {'edges': [list(e) for e in edges], 'cycles': final_cycles}

# EVOLVE-BLOCK-END
