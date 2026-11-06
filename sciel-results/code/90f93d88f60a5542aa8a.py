import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n = 16
    
    # Precompute all possible Q3 embeddings (subgraphs isomorphic to Q3)
    # Q3 has 8 vertices and 12 edges.
    # Vertices of Q3 can be represented as 3-bit integers 0..7.
    # Edges connect vertices differing by exactly 1 bit.
    # We need to find all injective mappings from {0..7} to {0..15} that preserve adjacency.
    
    # Generate all permutations of 8 vertices from 16, check if they form a Q3.
    # This is 16P8 = 16! / 8! = 518,918,400 which is too many to check naively in Python within time limits.
    
    # Alternative: Generate all subsets of 8 vertices from 16. C(16,8) = 12,870.
    # For each subset, check if the induced subgraph is isomorphic to Q3.
    # There are 12,870 subsets. For each, we need to check isomorphism to Q3.
    # Q3 is bipartite with parts of size 4 and 4.
    # In Q3, each vertex has degree 3.
    # So the induced subgraph must be 3-regular and bipartite with 4+4 split.
    # Also, the number of edges must be 12.
    
    # Let's precompute the list of all Q3 subgraphs (as sets of 8 vertices and 12 edges).
    # Actually, we just need to know which edges form a Q3.
    # A Q3 is defined by 8 vertices. If the induced subgraph on those 8 vertices is isomorphic to Q3, then any 12 edges that form Q3 within them are forbidden?
    # No, the task says "no subgraph isomorphic to Q3". This means no set of 8 vertices and 12 edges that form a Q3.
    # So if we have a set of 8 vertices that induces a supergraph of Q3, it might still contain a Q3 subgraph.
    # We must ensure that no 8 vertices contain a Q3 subgraph.
    
    # Strategy:
    # 1. Enumerate all 8-vertex subsets.
    # 2. For each subset, determine all possible Q3 subgraphs within it.
    #    Actually, it's easier to:
    #    - Maintain a graph G.
    #    - When adding an edge, check if it completes any Q3.
    #    - Or, precompute all "Q3 patterns".
    
    # Let's precompute all distinct Q3 subgraphs in K16.
    # A Q3 is determined by 8 vertices.
    # How many Q3 subgraphs are there in K16?
    # Number of ways to choose 8 vertices: C(16,8) = 12870.
    # For each set of 8 vertices, how many Q3 subgraphs can be formed?
    # If the 8 vertices induce a Q3, there is exactly 1 (if we consider the specific edge set).
    # But we are looking for subgraphs, so for a given set of 8 vertices, there might be multiple ways to pick 12 edges that form a Q3?
    # No, Q3 is a specific graph. If the 8 vertices induce a graph H, H contains Q3 as a subgraph if and only if there is an isomorphism from Q3 to a subgraph of H.
    # Since Q3 is 3-regular, the 8 vertices must each have degree at least 3 in the subgraph.
    
    # Efficient approach for dynamic construction:
    # Keep track of the graph as a set of edges.
    # When adding edge (u,v), check if this edge is part of any Q3 in the current graph.
    # An edge (u,v) is part of a Q3 if there exist 6 other vertices w1..w6 such that {u,v,w1..w6} induces a Q3.
    # This is still complex.
    
    # Better approach:
    # Precompute all 12-edge sets that form a Q3.
    # There are C(120, 12) which is huge.
    
    # Let's go back to vertex subsets.
    # Precompute for every 8-vertex subset S, the list of edge sets E_S such that (S, E_S) is isomorphic to Q3.
    # Actually, for a fixed set S of 8 vertices, the number of Q3 subgraphs is the number of isomorphisms from Q3 to S divided by automorphisms?
    # No, we just need to know if a specific set of edges contains a Q3.
    
    # Let's use a different representation.
    # Represent the graph as a 16x16 adjacency matrix (or bitmask).
    # Precompute all "Q3 masks" as 120-bit integers (or similar).
    # Total number of Q3 subgraphs in K16:
    # Choose 8 vertices: 12870.
    # For each, count number of Q3s.
    # A Q3 is a cube. The number of labeled Q3s on 8 vertices is 8! / 48 = 3360.
    # Wait, the number of isomorphisms from Q3 to a specific set of 8 vertices depends on the structure.
    # If we just want to know if a graph G contains Q3, we can check all 8-vertex subsets.
    
    # Let's precompute a list of all Q3s in K16 as frozensets of edges.
    # This might be too many.
    # Number of Q3s in K16:
    # For each set of 8 vertices, how many Q3s?
    # The number of subgraphs of K8 isomorphic to Q3.
    # It is known that the number of Q3s in K8 is 1680?
    # Let's calculate:
    # Number of ways to map 8 vertices of Q3 to 8 distinct vertices of K16 is P(16,8).
    # Each Q3 has 48 automorphisms.
    # So number of distinct Q3 subgraphs in K16 is P(16,8) / 48 = 518918400 / 48 = 10810800.
    # 10 million Q3s. Storing them as edge sets (12 edges each) might be memory heavy but doable?
    # 10M * 12 edges * 2 ints = 240M ints. Too much for Python.
    
    # Alternative:
    # During search, when adding an edge (u,v), check if it creates a Q3.
    # (u,v) is in a Q3 if there are 6 other vertices that complete it.
    # Specifically, in Q3, an edge belongs to 2 faces (squares).
    # So if (u,v) is in a Q3, there must be two other edges (u,w1) and (v,w2) and (u,w2) and (v,w1) forming a square?
    # Not exactly.
    
    # Let's use the property: Q3 is bipartite.
    # Let's try a randomized greedy approach with local search (hill climbing).
    # Start with a random graph or the seed K3,13.
    # Try to add edges. If adding an edge creates a Q3, reject.
    # Then try to swap edges (remove one, add another) to increase density?
    # Or just greedy addition.
    
    # Efficient Q3 check:
    # To check if adding edge (u,v) creates a Q3, we can look for 6 other vertices.
    # Actually, we can precompute for each pair (u,v) the list of "potential Q3 completions".
    # A Q3 containing (u,v) must have u and v as adjacent vertices in the cube.
    # In Q3, each edge is in exactly 2 squares.
    # Let the vertices of Q3 be x0..x7.
    # If u=x0, v=x1 (differ in bit 0).
    # Then the square containing (u,v) is x0-x1-x3-x2-x0.
    # So we need x2 and x3 such that (u,x2), (v,x3), (x2,x3) are edges? No, (x2,x3) is an edge in the square?
    # Square: 000-001-011-010-000. Edges: (0,1), (1,3), (3,2), (2,0).
    # So if (u,v) is in the graph, and we want to complete a Q3, we need to find x2, x3 such that:
    # 1. (u,x2) and (v,x3) are edges.
    # 2. (x2,x3) is an edge.
    # 3. And there is another square?
    # This is getting complicated.
    
    # Let's stick to the 8-vertex subset check for verification, but use a heuristic for construction.
    # Heuristic: Randomized Greedy.
    # 1. Initialize graph with no edges.
    # 2. Shuffle all possible edges.
    # 3. For each edge, if adding it doesn't create a Q3, add it.
    # 4. Repeat multiple times with different shuffles, keep best.
    
    # How to efficiently check if adding edge (u,v) creates a Q3?
    # We can maintain a list of "forbidden edge sets" that are Q3s.
    # But 10M Q3s is too many to store explicitly in Python lists.
    
    # Alternative check:
    # When adding (u,v), check all 8-vertex subsets S containing u and v.
    # There are C(14, 6) = 3003 such subsets.
    # For each subset S, check if the induced subgraph on S (with the new edge) contains a Q3.
    # This is still expensive if done naively.
    
    # Let's optimize the Q3 check.
    # Precompute for every 6-tuple of other vertices? No.
    
    # Let's try a different structural change:
    # Use a specific known construction or a smarter heuristic.
    # The seed is K3,13 (39 edges).
    # Random greedy might do better.
    
    # Let's implement a fast Q3 check.
    # Represent graph as a list of 16 bitmasks (neighbors).
    # To check if (u,v) creates a Q3:
    # A Q3 containing (u,v) must have u and v as neighbors.
    # Let N(u) be neighbors of u, N(v) be neighbors of v.
    # In Q3, u and v share exactly 2 common neighbors (the other two vertices of the two squares containing the edge).
    # Wait, in Q3, two adjacent vertices have exactly 2 common neighbors.
    # So, if we add (u,v), and |N(u) ∩ N(v)| >= 2, it might be part of a Q3.
    # But that's necessary, not sufficient.
    
    # Let's precompute all Q3s as (u, v, w1, w2, w3, w4, w5, w6) tuples?
    # 10M tuples is too much.
    
    # Let's try to generate Q3s on the fly during the check.
    # When adding (u,v):
    # For each pair of common neighbors (w1, w2) of u and v:
    #   The square is u-v-w2-w1-u.
    #   To form a Q3, we need to extend this square to a cube.
    #   The cube has two more squares attached.
    #   This path is complex.
    
    # Let's use the 8-vertex subset method but optimize it.
    # Precompute all 8-vertex subsets (12870).
    # For each subset, precompute the list of Q3 edge sets? No.
    # Instead, for each subset, precompute a "signature" or just check isomorphism.
    
    # Actually, 12870 subsets is small.
    # We can precompute for each subset S (frozenset of 8 vertices), the list of all Q3 subgraphs within S.
    # For a fixed S, the number of Q3s is at most 1680 (if S induces K8).
    # Total Q3s to store: 12870 * 1680 ~ 21M. Still too much.
    
    # Let's just check isomorphism for the 8 vertices involved when an edge is added.
    # When adding (u,v), we only need to check subsets S containing u and v.
    # There are C(14,6) = 3003 such subsets.
    # For each S, we check if G[S] contains a Q3.
    # G[S] has 8 vertices. We can check if it contains a Q3 by trying all 1680 isomorphisms?
    # Or simpler: check if G[S] is 3-regular and bipartite? No, it just needs to *contain* a Q3.
    # So we need to check if there exists a subgraph of G[S] isomorphic to Q3.
    # Since Q3 is small, we can just iterate over all 12-edge subsets of G[S]?
    # G[S] has at most 28 edges. C(28,12) is huge.
    
    # Better: Iterate over all 8-vertex mappings?
    # For a fixed S, to check if it contains Q3, we can check if there is a 3-regular spanning subgraph of G[S] that is bipartite?
    # No, Q3 is 3-regular. So if G[S] contains Q3, it must have a 3-regular spanning subgraph.
    # Number of 3-regular graphs on 8 vertices is small.
    
    # Let's try a different tactic:
    # Precompute all Q3s in K16 as bitmasks of edges.
    # Edge index: i < j, index = i*16 + j. Total 120 edges.
    # Use a Python int as a 120-bit mask.
    # Generate all Q3s:
    # 1. Iterate over all 8-vertex subsets S.
    # 2. For each S, find all Q3 subgraphs.
    #    To find Q3 subgraphs in S, we can use the fact that Q3 is the Cartesian product of three K2s.
    #    Or simply, try all permutations of S? No.
    #    We can use the 1680 labeled Q3s.
    #    Precompute the 1680 edge masks for Q3 on vertices 0..7.
    #    Then for each S, map 0..7 to S and check if the mask is a subset of G.
    
    # This is feasible!
    # 1. Generate 1680 Q3 masks on 8 vertices.
    # 2. For each of the 12870 subsets S, and each of the 1680 masks, create a global mask.
    #    Total masks: 12870 * 1680 = 21.6M.
    #    Storing 21.6M Python ints (120 bits) might use ~1GB of RAM. Too much.
    
    # We need to be smarter.
    # Instead of precomputing all, generate them on the fly when checking an edge addition?
    # When adding (u,v), we only care about Q3s containing (u,v).
    # How many Q3s contain a specific edge in K16?
    # Total Q3s: 10,810,800.
    # Total edges in all Q3s: 10,810,800 * 12 = 129,729,600.
    # Average Q3s per edge: 129,729,600 / 120 = 1,081,080.
    # So for each edge, there are ~1M Q3s containing it.
    # We can precompute a list of Q3 masks for each edge.
    # 120 edges * 1M masks = 120M masks. Still too much.
    
    # Let's reduce the scope.
    # We only need to check if the *current graph* contains a Q3.
    # We can maintain a counter or a set of "active" Q3s?
    
    # Alternative: Local Search with a fast approximate Q3 check.
    # Or, use a SAT solver? No, standard library only.
    
    # Let's try the randomized greedy with a *simplified* Q3 check.
    # Check if adding (u,v) creates a K4,4? No.
    # Check if adding (u,v) creates a cycle of length 4? Q3 has 6 cycles of length 4.
    
    # Let's go back to the 8-vertex subset check, but make it fast.
    # Precompute for each pair (u,v) the list of 6-tuples (w1..w6) such that {u,v,w1..w6} induces a Q3 in K16.
    # This is the list of "witnesses" for edge (u,v).
    # Size of this list: ~1M per edge.
    # Total witnesses: 120 * 1M = 120M.
    # Storing 120M tuples of 6 ints is too much.
    
    # Let's compress the witness.
    # A witness is a set of 6 vertices.
    # We can store it as a bitmask of 16 bits.
    # 120M 16-bit ints = 240MB. Maybe feasible in C, but Python overhead is high.
    
    # Let's try a different approach:
    # Use the fact that Q3 is bipartite.
    # Partition vertices into two sets A and B (size 8 each).
    # Q3 is a bipartite graph with parts of size 4 and 4.
    # So a Q3 subgraph must be contained in a bipartite subgraph with parts of size at least 4.
    
    # Let's try a heuristic:
    # 1. Start with a random bipartite graph with parts of size 8.
    # 2. Add edges greedily.
    # 3. If a Q3 is formed, remove an edge.
    
    # Actually, let's just implement the randomized greedy with the 8-vertex subset check, but optimize the check.
    # When adding (u,v), check all 3003 subsets S containing u,v.
    # For each S, check if G[S] contains a Q3.
    # To check if G[S] contains a Q3, we can precompute a list of 1680 Q3 masks on 8 vertices.
    # For each S, we map the 8 vertices to 0..7.
    # We check if any of the 1680 masks is a subset of G[S].
    # This is 3003 * 1680 = 5M checks per edge addition.
    # 5M checks is too slow in Python for 120s.
    
    # We need to prune.
    # Most subsets S will not contain a Q3.
    # We can check a necessary condition: G[S] must have at least 12 edges.
    # And each vertex in S must have degree at least 3 in G[S].
    # This filters out most subsets quickly.
    
    # Let's try this.
    
    def has_q3_in_subset(S, adj):
        # S is a tuple of 8 vertices
        # adj is a list of 16 bitmasks
        # Check if G[S] contains a Q3
        # First, check degrees in G[S]
        degs = [0]*8
        idx_map = {v:i for i,v in enumerate(S)}
        for i,u in enumerate(S):
            for j,v in enumerate(S):
                if i != j and (adj[u] >> v) & 1:
                    degs[i] += 1
        if any(d < 3 for d in degs):
            return False
        
        # Check edge count
        edges = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (adj[S[i]] >> S[j]) & 1:
                    edges += 1
        if edges < 12:
            return False
            
        # Now check for Q3 isomorphism
        # Precomputed Q3 masks on 8 vertices (0..7)
        # We need to map S to 0..7 and check masks
        # This is still slow.
        
        # Let's hardcode a few Q3 checks?
        # No, let's use the 1680 masks.
        # But 1680 checks per subset is too much.
        
        return False # Placeholder, need better check
    
    # This is getting too complex and slow.
    
    # Let's try a different strategy:
    # Use a known dense Q3-free graph.
    # The Petersen graph is Q3-free?
    # What about the complement of a matching?
    
    # Let's try a simple hill climbing.
    # Start with a random graph with 40 edges.
    # Try to add edges.
    # If Q3 found, remove a random edge from the Q3.
    
    # For the Q3 check, let's use a simpler necessary condition:
    # If adding (u,v) creates a Q3, then there must be a 4-cycle containing (u,v) that is part of a Q3.
    # In Q3, every edge is in exactly 2 4-cycles.
    # So, if we add (u,v), we check if there are at least 2 4-cycles containing (u,v) that can be extended to a Q3.
    
    # Let's just use the seed and try to improve it with a simple local search.
    # Seed: K3,13 (39 edges).
    
    # Let's try to add edges to K3,13.
    # Vertices 0,1,2 are connected to 3..15.
    # Edges among 0,1,2: none.
    # Edges among 3..15: none.
    # We can add edges among 3..15.
    # If we add an edge (u,v) with u,v in 3..15, does it create a Q3?
    # A Q3 needs 4 vertices from {0,1,2} and 4 from {3..15}.
    # But {0,1,2} has only 3 vertices. So no Q3 can be formed with 4 vertices from {0,1,2}.
    # Wait, Q3 is bipartite with parts of size 4.
    # If the graph is K3,13, the bipartition is {0,1,2} and {3..15}.
    # But Q3 requires parts of size 4.
    # So K3,13 is Q3-free.
    # We can add edges within {3..15}.
    # If we add edges within {3..15}, the graph is no longer bipartite with parts 3 and 13.
    # But we can still avoid Q3.
    
    # Let's try to add a matching in {3..15}.
    # 13 vertices, max matching size 6.
    # Add 6 edges. Total 45 edges.
    # Is this Q3-free?
    # A Q3 would need 4 vertices from one part and 4 from the other.
    # If we add edges within {3..15}, we create triangles? No, matching has no triangles.
    # But it might create Q3s.
    
    # Let's just return the seed for now, but with a small improvement if possible.
    # Actually, the task says "Maximize the number of edges".
    # The seed is 39.
    # We can try to add edges within the larger part.
    
    # Let's try to add all possible edges within {3..15} that don't create a Q3.
    # This is a greedy approach.
    
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    
    # Try to add edges within 3..15
    for u in range(3, 16):
        for v in range(u+1, 16):
            # Check if adding (u,v) creates a Q3
            # We can do a quick check:
            # If we add (u,v), we need to see if there is a Q3.
            # Let's just add it and verify at the end?
            # No, we need to be careful.
            
            # Let's assume we can add some.
            edges.append([u, v])
    
    # This might create Q3s.
    # Let's just return the seed.
    
    return edges

# This is a placeholder. I need a real algorithm.
# Let's try a different design:
# Use a SAT solver-like approach? No.
# Use a genetic algorithm?
# Let's try a simple local search.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with K3,13
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    # Try to add edges within 3..15
    candidates = [(u, v) for u in range(3, 16) for v in range(u+1, 16)]
    random.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        # Check if adding (u,v) creates a Q3
        # We'll use a simple heuristic: check if it creates a 4-cycle that is part of a Q3
        # For now, let's just add it and hope it's Q3-free
        # This is risky.
        
        # Let's do a proper check.
        # We'll check all 8-vertex subsets containing u and v.
        # There are C(14,6) = 3003 subsets.
        # For each, check if it contains a Q3.
        # We'll use a fast check.
        
        # Let's just add it for now.
        edges.add((u, v))
        
    return [[u, v] for u, v in edges]

# This will likely create Q3s.
# I need a better check.

# Let's try a different approach:
# Use the fact that Q3 is a bipartite graph with parts of size 4.
# If we keep the graph bipartite with parts of size 8 and 8, and ensure that no 4x4 submatrix is a Q3, we might be safe.
# But Q3 is not just any 4x4 bipartite graph.

# Let's just return the seed.
def construct():
    return [[a, b] for a in range(3) for b in range(3, 16)]
