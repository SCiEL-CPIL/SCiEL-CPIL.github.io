"""
Strategy: Structural Construction based on PG(2,5) Polarity Graph.
The graph with 31 vertices and 90 edges minimizing C4s is the polarity graph of PG(2,5),
which is 6-regular. We construct this base graph explicitly.
Then, we need to add 1 edge to reach 91 edges.
We search for the edge addition that minimizes the total number of C4s.
Since the base graph is 6-regular, adding any edge (u, v) where u and v are non-adjacent
will create new C4s. The number of new C4s is determined by the common neighbors of u and v
in the base graph, plus interactions with the new edge.
Actually, the C4 count is global. We will construct the base graph, then try all possible
non-edges to add, and pick the one that results in the minimum total C4 count.
If there are ties, we might need to adjust local structure, but likely one edge is best.
To be robust, we also allow small local swaps (2-flips) after adding the edge to see if
we can reduce C4s further by rearranging the "excess" degree.
"""
import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 145  # Safety margin
    
    # 1. Construct the Polarity Graph of PG(2,5)
    # PG(2,5) has points represented by homogeneous coordinates [x:y:z] over GF(5).
    # There are (5^3 - 1) / (5 - 1) = 124/4 = 31 points.
    # We list all points.
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                # Since we are in GF(5), we can just iterate and normalize.
                # Simpler: iterate all non-zero triples, divide by first non-zero.
                # But to avoid duplicates, we fix the first non-zero to 1.
                # Case 1: x != 0. y, z in 0..4. (5 * 5 * 5 = 125? No.)
                # If x != 0, we can scale so x=1. y, z free. 25 points.
                # If x == 0, y != 0. Scale so y=1. z free. 5 points.
                # If x == 0, y == 0, z != 0. Scale so z=1. 1 point.
                # Total 25 + 5 + 1 = 31.
                pass
                
    # Let's generate the 31 points explicitly.
    pts = []
    # Type 1: [1, y, z]
    for y in range(5):
        for z in range(5):
            pts.append((1, y, z))
    # Type 2: [0, 1, z]
    for z in range(5):
        pts.append((0, 1, z))
    # Type 3: [0, 0, 1]
    pts.append((0, 0, 1))
    
    # Verify count
    assert len(pts) == 31
    
    # The polarity graph ER(31, 5) is defined by:
    # Two points P=[x1,y1,z1] and Q=[x2,y2,z2] are adjacent iff x1*x2 + y1*y2 + z1*z2 == 0 in GF(5).
    # Note: This is a self-adjacent graph in the geometric sense (polarity), but as a simple graph,
    # we usually take the graph where edges exist for distinct points satisfying the condition.
    # Wait, the standard Erdős-Rényi graph ER(n, q) for n=q^2+q+1 is defined on the points of PG(2,q).
    # Two points are adjacent if they are orthogonal with respect to the polarity x1x2+y1y2+z1z2=0.
    # For q=5 (odd), the graph is regular of degree q+1 = 6.
    # Let's build the adjacency matrix.
    
    adj_base = np.zeros((n, n), dtype=np.int8)
    
    def dot_product(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
    
    for i in range(n):
        for j in range(i+1, n):
            if dot_product(pts[i], pts[j]) == 0:
                adj_base[i, j] = 1
                adj_base[j, i] = 1
                
    # Count edges in base graph
    edge_count_base = np.sum(adj_base) // 2
    # It should be 31 * 6 / 2 = 93? 
    # Wait. Degree is q+1 = 6. Total edges = 31*6/2 = 93.
    # But ex(31, C4) = 90. 
    # Let me re-verify the theorem.
    # The number of edges in ER(n, q) is n(q+1)/2.
    # For q=5, n=31, edges = 31*6/2 = 93.
    # However, the problem states ex(31, C4) = 90.
    # This implies the polarity graph has 93 edges, which is MORE than 90.
    # So the polarity graph is NOT the extremal graph for C4-free?
    # Actually, ex(n, C4) is the maximum number of edges with NO C4.
    # The task is to minimize the NUMBER of C4s, not to be C4-free.
    # The "extremal" graph for minimizing C4s given a fixed number of edges M is different.
    # For M=91, we are just above the C4-free threshold (90).
    # A graph with 90 edges can have 0 C4s.
    # A graph with 91 edges MUST have at least 1 C4?
    # If we take a C4-free graph with 90 edges and add 1 edge, we create some C4s.
    # The polarity graph has 93 edges and many C4s. It is not the right base.
    
    # Correction:
    # The goal is to MINIMIZE C4s.
    # Start with a C4-free graph with 90 edges.
    # Add 1 edge. Minimize the number of C4s created.
    # The C4-free graph with 90 edges on 31 vertices is likely related to the projective plane structure but not the full polarity graph.
    # Actually, the Reiman bound says ex(n, C4) <= n/4(1+sqrt(4n-3)).
    # For n=31, n/4(1+sqrt(121)) = 31/4(1+11) = 31/4(12) = 93.
    # So 90 is below the bound. It is known that ex(31, C4) = 90.
    # There exist C4-free graphs with 90 edges.
    # We need to find one, add an edge, and count C4s.
    
    # Strategy:
    # 1. Find a C4-free graph with 90 edges.
    #    A good candidate is a subgraph of the polarity graph? No, polarity graph has 93 edges and C4s.
    #    Actually, the polarity graph of PG(2, q) for odd q has C4s.
    #    The C4-free graphs are often constructed via finite fields differently or by removing edges from dense structures.
    #    
    #    Alternative: Use a random C4-free graph generation.
    #    We can start with a random graph and remove edges to break C4s until we have 90 edges and 0 C4s.
    #    Then add 1 edge and count C4s.
    #    To minimize C4s after adding, we should pick the edge that creates the fewest C4s.
    #    The number of C4s created by adding edge (u, v) to a C4-free graph is exactly the number of paths of length 3 between u and v?
    #    No. A C4 is u-a-v-b-u. If we add edge (u, v), a new C4 is formed for every pair of common neighbors (a, b) of u and v?
    #    No, if the graph is C4-free, there are NO pairs of common neighbors for any non-adjacent pair?
    #    Wait. If G is C4-free, then any two non-adjacent vertices have at most 1 common neighbor.
    #    If they had 2 common neighbors a, b, then u-a-v-b-u would be a C4.
    #    So in a C4-free graph, for any non-edge (u, v), common_neighbors(u, v) <= 1.
    #    Therefore, adding edge (u, v) creates:
    #    - If common_neighbors(u, v) == 0: No C4s created involving this edge directly as a "chord"?
    #      Wait. A C4 containing the new edge (u, v) must be u-v-a-b-u or u-v-b-a-u.
    #      This requires a path of length 3 between u and v in the original graph.
    #      Path u-a-v-b? No.
    #      Cycle: u - v - a - b - u.
    #      Edges: (u, v) [new], (v, a) [old], (a, b) [old], (b, u) [old].
    #      So we need a path of length 3 from u to v in the old graph: u-b-a-v.
    #      The number of such paths is the number of C4s created.
    #      
    #    So, to minimize C4s, we want to add an edge (u, v) such that the number of length-3 paths between u and v in the C4-free base graph is minimized.
    #    Ideally, we want a pair (u, v) with 0 length-3 paths.
    #    If such a pair exists, the resulting graph has 0 C4s?
    #    Wait, if the base graph is C4-free, and we add an edge that creates 0 C4s, the new graph is also C4-free.
    #    But ex(31, C4) = 90. So a graph with 91 edges CANNOT be C4-free.
    #    This implies that in ANY C4-free graph with 90 edges, EVERY non-edge (u, v) has at least 1 length-3 path between u and v.
    #    So the minimum C4s will be at least 1.
    #    We want to find the configuration where the minimum number of length-3 paths for any non-edge is as small as possible (ideally 1).
    
    # Plan:
    # 1. Generate a C4-free graph with 90 edges.
    # 2. For every non-edge (u, v), calculate the number of length-3 paths between u and v.
    # 3. Pick the non-edge with the minimum count (likely 1).
    # 4. Add it. Return the edges.
    # 5. To be safe, we can also try local swaps (2-flips) to see if we can find a 91-edge graph with fewer C4s by moving away from the "C4-free + 1" structure, but given ex(31)=90, the number of C4s must be small.
    
    # Step 1: Find a C4-free graph with 90 edges.
    # We can use a greedy construction or random restarts with C4 removal.
    # Given the small size, we can use a fast heuristic.
    
    rng = random.Random(42)
    
    def count_c4_fast(adj_mat):
        A = adj_mat.astype(np.int32)
        common = A @ A
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        non_edge_mask = mask & (adj_mat == 0)
        cv = common[non_edge_mask]
        total = np.sum(cv * (cv - 1) // 2)
        return total // 2
        
    def find_c4_free_graph():
        # Start with random graph, remove edges to break C4s until 0 C4s and 90 edges.
        # Or start with 90 edges and fix C4s.
        # Let's try: Start with random 90-edge graph.
        # While C4s > 0:
        #   Find a C4. Remove one edge from it. Add a random edge to maintain count?
        #   No, just remove edge. If edges < 90, add random non-edge that doesn't create C4?
        #   This is tricky.
        
        # Better: Start with a known sparse structure.
        # The polarity graph has 93 edges. If we remove 3 edges, do we get a C4-free graph?
        # Let's check the C4 count of the polarity graph.
        
        adj_test = adj_base.copy()
        c4_count = count_c4_fast(adj_test)
        
        # If the polarity graph has few C4s, we can remove edges to kill them.
        # If we remove an edge from a C4, we break that C4.
        # We need to remove 3 edges to get to 90.
        # If we can choose 3 edges such that their removal breaks ALL C4s, we have a C4-free graph with 90 edges.
        
        if c4_count == 0:
            return adj_test # Unlikely for 93 edges
        
        # Let's find a set of edges to remove.
        # Greedy: Find a C4, remove one of its edges. Repeat.
        # We need to remove exactly 3 edges.
        
        best_adj = None
        best_c4 = float('inf')
        
        # Try removing all combinations of 3 edges from the polarity graph?
        # Number of edges is 93. C(93, 3) is ~ 130,000.
        # 130,000 * (time to count C4s) might be too slow if count is slow.
        # But count_c4_fast is vectorized. 130k * 1ms = 130s. Too slow.
        
        # Let's try a local search approach on the 93-edge graph to minimize C4s,
        # targeting 90 edges.
        
        # Actually, let's just use a random restart search for a C4-free graph with 90 edges.
        
        for attempt in range(50):
            if time.time() > deadline - 20: break
            
            # Start with random 90 edges
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edges = rng.sample(all_pairs, 90)
            adj = np.zeros((n, n), dtype=np.int8)
            for u, v in edges:
                adj[u, v] = 1
                adj[v, u] = 1
                
            c4s = count_c4_fast(adj)
            
            # Hill climbing to reduce C4s
            # Move: Swap an edge in a C4 with a non-edge.
            # If we swap an edge (u, v) that is part of a C4 with a non-edge (a, b),
            # we might reduce C4s.
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                c4s = count_c4_fast(adj)
                if c4s == 0:
                    break
                
                # Find a C4
                # Get common neighbors
                A = adj.astype(np.int32)
                common = A @ A
                # Find a pair (u, v) non-adjacent with common >= 2
                # Or adjacent? C4s are counted on non-adjacent pairs in the formula?
                # No, C4 u-v-w-x-u. u and w are non-adjacent? No, u and w are opposite.
                # In the formula sum C(common[u][v], 2) over non-edges,
                # a C4 contributes 1 to the count for the pair of opposite vertices.
                # So if common[u][v] >= 2 and adj[u][v]==0, there is at least one C4.
                
                # Find any non-edge with common >= 2
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                non_edge_mask = mask & (adj == 0)
                cv = common[non_edge_mask]
                indices = np.where(cv >= 2)
                
                if len(indices[0]) == 0:
                    break # No C4s? But c4s > 0? 
                    # This can happen if the C4s are "degenerate" or if my formula is slightly off?
                    # No, the formula is correct.
                    
                # Pick a random C4-defining pair
                idx = rng.randint(0, len(indices[0]) - 1)
                u, v = indices[0][idx], indices[1][idx]
                
                # The C4s involve u, v and 2 common neighbors.
                # Common neighbors of u and v:
                cn = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                
                # We want to break a C4.
                # We can remove an edge (u, cn[0]) or (v, cn[0]) or (cn[0], cn[1])?
                # Removing (u, cn[0]) breaks all C4s using that edge.
                # But we must maintain 90 edges. So we must add another edge.
                # Add a random non-edge (a, b).
                
                # Heuristic: Remove an edge incident to u or v, add a random non-edge.
                # Try a few swaps.
                
                best_swap_c4 = c4s
                best_swap = None
                
                # Candidate edges to remove: (u, cn[0]), (u, cn[1]), (v, cn[0]), (v, cn[1])
                candidates_to_remove = [(u, cn[0]), (u, cn[1]), (v, cn[0]), (v, cn[1])]
                
                for ru, rv in candidates_to_remove:
                    if ru > rv: ru, rv = rv, ru
                    if adj[ru, rv] == 0: continue
                    
                    # Try adding a random non-edge
                    for _ in range(10):
                        a = rng.randint(0, n-1)
                        b = rng.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if adj[a, b] == 1: continue
                        if (a, b) == (ru, rv): continue
                        
                        # Perform swap
                        adj[ru, rv] = 0
                        adj[rv, ru] = 0
                        adj[a, b] = 1
                        adj[b, a] = 1
                        
                        new_c4s = count_c4_fast(adj)
                        
                        if new_c4s < best_swap_c4:
                            best_swap_c4 = new_c4s
                            best_swap = (ru, rv, a, b)
                            
                        # Revert
                        adj[ru, rv] = 1
                        adj[rv, ru] = 1
                        adj[a, b] = 0
                        adj[b, a] = 0
                        
                if best_swap:
                    ru, rv, a, b = best_swap
                    adj[ru, rv] = 0
                    adj[rv, ru] = 0
                    adj[a, b] = 1
                    adj[b, a] = 1
                    improved = True
                    
            if count_c4_fast(adj) == 0:
                return adj
                
        # If we didn't find a C4-free graph, return the best one found?
        # Or just return a random 91-edge graph?
        # Let's try the polarity graph minus 3 edges as a fallback.
        return None

    base_adj = find_c4_free_graph()
    
    if base_adj is None:
        # Fallback: Use polarity graph and remove 3 random edges
        adj = adj_base.copy()
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
        removed = rng.sample(edges_list, 3)
        for u, v in removed:
            adj[u, v] = 0
            adj[v, u] = 0
        base_adj = adj
        
    # Now, base_adj has 90 edges (hopefully 0 C4s).
    # We need to add 1 edge to get 91 edges.
    # We want to minimize C4s.
    # As analyzed, the number of new C4s is the number of length-3 paths between u and v.
    
    best_total_c4 = float('inf')
    best_final_adj = None
    
    # Calculate length-3 paths for all non-edges
    A = base_adj.astype(np.int32)
    # Number of paths of length 3 from u to v is (A^3)[u, v]
    # A^3 can be computed as A @ (A @ A)
    A2 = A @ A
    A3 = A @ A2
    
    # Iterate over non-edges
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if base_adj[i, j] == 0]
    
    for u, v in non_edges:
        # Number of C4s created by adding (u, v) is A3[u, v]
        # Note: A3[u, v] counts paths u-a-b-v.
        # Each such path forms a C4 u-v-b-a-u.
        # Are there duplicates?
        # A path u-a-b-v corresponds to the cycle u-v-b-a-u.
        # The reverse path v-b-a-u corresponds to the same cycle.
        # A3[u, v] counts directed paths.
        # Each undirected C4 contains 4 directed paths of length 3?
        # Let's check.
        # Cycle u-v-w-x-u.
        # Paths of length 3 from u to v:
        # 1. u-x-w-v
        # 2. u-v... no, u to v is length 1.
        # Wait. The path must be length 3 in the OLD graph.
        # Old graph does not have edge (u, v).
        # Path u-x-w-v. Edges (u, x), (x, w), (w, v).
        # This forms C4 u-v-w-x-u.
        # Is there another path from u to v?
        # u-w-x-v? No, (u, w) is not an edge in C4 u-v-w-x-u.
        # So for each C4 containing (u, v), there is exactly ONE path of length 3 from u to v in the rest of the graph?
        # Let's verify.
        # C4: 1-2-3-4-1. Add edge (1, 3).
        # Paths of length 3 from 1 to 3 in old graph (1-2-3-4-1):
        # 1-2-3 (length 2). No.
        # 1-4-3 (length 2). No.
        # 1-2-3-4-1... no.
        # Wait. A path of length 3 is a sequence of 3 edges.
        # 1-2-3 is 2 edges.
        # 1-4-3 is 2 edges.
        # So A3[1, 3] should be 0?
        # Let's re-read the C4 formation.
        # New edge (u, v).
        # C4: u-v-a-b-u.
        # Edges in old graph: (v, a), (a, b), (b, u).
        # This is a path from v to u of length 3: v-a-b-u.
        # So the number of C4s is the number of paths of length 3 from v to u (or u to v) in the old graph.
        # In the example 1-2-3-4-1, adding (1, 3).
        # Paths from 3 to 1 of length 3:
        # 3-2-1 (length 2).
        # 3-4-1 (length 2).
        # So 0 C4s?
        # But 1-2-3-4-1 is a C4. If I add (1, 3), I create triangles 1-2-3 and 1-3-4.
        # I do NOT create a new C4. The C4 was already there!
        # The task is to MINIMIZE the NUMBER of C4s.
        # If I start with a C4-free graph, I have 0 C4s.
        # If I add an edge (u, v), I create C4s that INCLUDE the new edge.
        # A C4 including the new edge (u, v) is u-v-a-b-u.
        # This requires a path of length 3 from u to v in the OLD graph?
        # No. u-v is the new edge.
        # The rest of the cycle is v-a-b-u.
        # This is a path from v to u of length 3 in the OLD graph.
        # So yes, the number of new C4s is the number of length-3 paths between u and v in the old graph.
        
        # In my example 1-2-3-4-1, the graph is NOT C4-free. It HAS a C4.
        # If the graph is C4-free, there are no cycles of length 4.
        # So if I add an edge, I can only create C4s if there are paths of length 3.
        # If the graph is C4-free, can it have paths of length 3?
        # Yes. A path of length 3 is not a cycle.
        # Example: 1-2-3-4. No C4.
        # Add edge (1, 4). Creates C4 1-2-3-4-1.
        # Path from 4 to 1 of length 3: 4-3-2-1. Yes.
        # So the logic holds.
        
        paths = A3[u, v]
        # Total C4s = 0 (from base) + paths
        if paths < best_total_c4:
            best_total_c4 = paths
            best_final_adj = base_adj.copy()
            best_final_adj[u, v] = 1
            best_final_adj[v, u] = 1
            
    if best_final_adj is None:
        # Fallback
        best_final_adj = base_adj.copy()
        # Add any non-edge
        for i in range(n):
            for j in range(i+1, n):
                if best_final_adj[i, j] == 0:
                    best_final_adj[i, j] = 1
                    best_final_adj[j, i] = 1
                    break
            else:
                continue
            break

    # Extract edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_final_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    # Recount
    A = adj.astype(np.int32)
    common = A @ A
    mask = np.triu(np.ones((31, 31), dtype=bool), k=1)
    non_edge_mask = mask & (adj == 0)
    cv = common[non_edge_mask]
    total = np.sum(cv * (cv - 1) // 2)
    print(f"C4: {total // 2}")
