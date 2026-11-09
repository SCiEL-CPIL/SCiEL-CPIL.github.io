import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Q3 has 8 vertices and 12 edges.
    # Vertices 0..7, edges between i,j if popcount(i^j)==1
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))

    def has_q3(adj, n=16):
        """Check if graph on n vertices with adjacency matrix adj contains Q3 as a subgraph.
        adj[i][j] is 1 if edge exists.
        We try all subsets of 8 vertices and all bijections to Q3 vertices.
        This is expensive, so we use pruning.
        """
        # Precompute candidate sets: for Q3, each vertex has degree 3.
        # Any Q3 subgraph must use 8 vertices each of degree >= 3 in the subgraph induced... 
        # Actually, in the host graph they just need to have the required edges.
        
        # Faster check: use bitmasks for adjacency
        # adj_mask[i] is a bitmask of neighbors of i
        
        for subset in itertools.combinations(range(n), 8):
            # Check if these 8 vertices could form Q3
            # We need to assign them to Q3 vertices 0..7 such that all Q3 edges are present.
            # Use backtracking with the 8 vertices.
            # Map subset indices to 0..7
            s = list(subset)
            
            # Pre-check: degrees in the induced subgraph must be at least the Q3 degrees (all 3)
            # Actually, the induced subgraph on s must contain Q3 as a subgraph, meaning 
            # the degree in the induced subgraph must be >= 3 for each vertex? No.
            # Q3 is 3-regular. If a vertex in the induced subgraph has degree < 3, it can't be part of Q3.
            # So all 8 vertices must have degree >= 3 in the induced subgraph.
            induced_degrees = [0] * 8
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[s[i]][s[j]]:
                        induced_degrees[i] += 1
                        induced_degrees[j] += 1
            if min(induced_degrees) < 3:
                continue
            
            # Now try to embed Q3 into this induced subgraph
            # Use backtracking: assign Q3 vertices 0..7 to s[0]..s[7]
            # q3_adj[q] is set of Q3 neighbors of q
            q3_adj = [set() for _ in range(8)]
            for (u, v) in q3_edges:
                q3_adj[u].add(v)
                q3_adj[v].add(u)
            
            # Order Q3 vertices by something? Just 0..7
            # Assignment array: assign[qi] = index in s (0..7)
            assign = [-1] * 8
            
            def backtrack(qi):
                if qi == 8:
                    return True
                for si in range(8):
                    if assign[si] == -1:  # si not used
                        # Check if s[si] can be assigned to Q3 vertex qi
                        ok = True
                        for qj in q3_adj[qi]:
                            if assign[qj] != -1:
                                sj = assign[qj]
                                if not adj[s[si]][s[sj]]:
                                    ok = False
                                    break
                        if ok:
                            assign[si] = qi
                            if backtrack(qi + 1):
                                return True
                            assign[si] = -1
                return False
            
            # We need to iterate over all assignments of Q3 vertices to s indices.
            # The above backtracking fixes Q3 vertex 0 to some s[si], etc.
            # Actually, the backtracking above tries to assign Q3 vertices 0..7 to s[0]..s[7]
            # but it's not quite right. Let's fix it.
            
            # Correct backtracking:
            # We assign each Q3 vertex q (0..7) to a position in s (0..7).
            # Let's use a different structure.
            
            def backtrack2():
                # Find first unassigned Q3 vertex
                for q in range(8):
                    if assign[q] == -1:
                        break
                else:
                    return True  # all assigned
                for si in range(8):
                    if assign[si] == -1:  # s[si] not used
                        # Check constraints with already assigned Q3 neighbors
                        ok = True
                        for q2 in q3_adj[q]:
                            if assign[q2] != -1:
                                si2 = assign[q2]
                                if not adj[s[si]][s[si2]]:
                                    ok = False
                                    break
                        if ok:
                            assign[q] = si
                            if backtrack2():
                                return True
                            assign[q] = -1
                return False
            
            # Reset assign
            assign = [-1] * 8
            if backtrack2():
                return True
        
        return False

    # Start with the known 48-edge graph
    edges = []
    # A = {0,1,2}, B = {3..15}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    edges.append([0, 1])
    edges.append([1, 2])
    edges.append([0, 2])
    edges.append([3, 4])
    edges.append([5, 6])
    edges.append([7, 8])
    edges.append([9, 10])
    edges.append([11, 12])
    edges.append([13, 14])

    best_edges = list(edges)
    best_count = len(edges)

    # Build adjacency matrix
    adj = [[0]*16 for _ in range(16)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1

    # Candidate edges to add: non-edges in B (vertices 3..15)
    # Currently B has a matching plus vertex 15 isolated.
    # We can try adding edges among B vertices.
    # Also possibly edges from A to B are already complete (K_{3,13}).
    # Edges within A are complete (K_3).
    # So the only missing edges are among B vertices that are not in the matching.
    
    # List all pairs in B
    b_vertices = list(range(3, 16))
    candidate_edges = []
    for i in range(len(b_vertices)):
        for j in range(i+1, len(b_vertices)):
            u = b_vertices[i]
            v = b_vertices[j]
            if adj[u][v] == 0:
                candidate_edges.append([u, v])

    # Try adding edges one by one, checking for Q3
    # We'll do a greedy/beam approach: try all single additions, keep the best valid ones.
    # Then try pairs, etc. But time is limited.
    
    # First, try all single edge additions
    valid_additions = []
    for ce in candidate_edges:
        u, v = ce
        if adj[u][v] == 1:
            continue
        adj[u][v] = 1
        adj[v][u] = 1
        # Check for Q3
        # This is expensive. We only have 130s.
        # has_q3 is very expensive. Let's see how long it takes.
        # For 16 vertices, C(16,8) = 12870 subsets. For each, backtracking.
        # This might be too slow to call many times.
        
        # Instead, let's be smarter. Only check subsets that contain u and v.
        # Actually, a new Q3 must use the new edge (u,v) or be created by it.
        # So we only need to check 8-sets that contain both u and v.
        # Number of such sets: C(14,6) = 3003. Still a lot, but better.
        
        # Even better: the new Q3 must use edge (u,v). So we check if there's a Q3 containing (u,v).
        # This means u and v must be adjacent in Q3. In Q3, adjacent vertices differ by 1 bit.
        # So we need to map u and v to two Q3 vertices that are adjacent.
        
        # Let's implement a targeted check.
        def has_q3_with_edge(adj, u, v, n=16):
            # Check if there's a Q3 subgraph that uses edge (u,v).
            # In Q3, let's say we map u->0, v->1 (since 0 and 1 are adjacent in Q3).
            # Then we need to find 6 more vertices to complete Q3.
            # The Q3 vertices are 0..7.
            # 0's neighbors in Q3: 1, 2, 4
            # 1's neighbors in Q3: 0, 3, 5
            # Since 0 and 1 are mapped to u and v, we need:
            # - 2 Q3 vertices that map to neighbors of 0 (excluding 1): {2, 4}
            # - 2 Q3 vertices that map to neighbors of 1 (excluding 0): {3, 5}
            # - 2 Q3 vertices that map to {6, 7}
            
            # Let's enumerate all ways to assign Q3 vertices to graph vertices.
            # Fixed: q0->u, q1->v
            # We need to assign q2,q3,q4,q5,q6,q7 to 6 distinct vertices from the remaining 14.
            # Constraints:
            # q2 adj q0 (so q2 must be neighbor of u), q2 not adj q1
            # q4 adj q0 (so q4 must be neighbor of u), q4 not adj q1
            # q3 adj q1 (so q3 must be neighbor of v), q3 not adj q0
            # q5 adj q1 (so q5 must be neighbor of v), q5 not adj q0
            # q6 adj q2, q4, q7
            # q7 adj q3, q5, q6
            # q2 adj q6, q4, q0
            # q4 adj q2, q6, q0
            # q3 adj q5, q7, q1
            # q5 adj q3, q7, q1
            # q6 adj q2, q4, q7
            # q7 adj q3, q5, q6
            
            # Let's get candidate sets:
            # C2 = neighbors of u in graph, excluding v
            # C3 = neighbors of v in graph, excluding u
            # Then q2, q4 must be in C2
            # q3, q5 must be in C3
            
            c2 = [w for w in range(n) if adj[u][w] and w != v]
            c3 = [w for w in range(n) if adj[v][w] and w != u]
            
            # q2 and q4 are distinct vertices in c2
            # q3 and q5 are distinct vertices in c3
            # Also, q2, q4, q3, q5 must all be distinct (they map to distinct Q3 vertices)
            
            for x2 in c2:
                for x4 in c2:
                    if x2 == x4:
                        continue
                    # q2=x2, q4=x4
                    # q2 and q4 must not be adjacent to v (q1)
                    if adj[x2][v] or adj[x4][v]:
                        continue
                    for x3 in c3:
                        for x5 in c3:
                            if x3 == x5:
                                continue
                            # q3 and q5 must not be adjacent to u (q0)
                            if adj[x3][u] or adj[x5][u]:
                                continue
                            # All 6 must be distinct
                            used = {u, v, x2, x4, x3, x5}
                            if len(used) < 6:
                                continue
                            # Now find x6, x7
                            # x6 must be neighbor of x2, x4, and x7
                            # x7 must be neighbor of x3, x5, and x6
                            # Also x6, x7 distinct from used
                            # x6 must be in c2? No. x6 is adjacent to x2 and x4, but not necessarily to u.
                            # In Q3, q6 is not adjacent to q0. So x6 not adj u.
                            # q6 is adjacent to q2, q4, q7.
                            # q7 is adjacent to q3, q5, q6.
                            # q7 is not adjacent to q1 (v). So x7 not adj v.
                            
                            # Candidates for x6: neighbors of x2 and x4, not in used, not adj u, not adj v?
                            # Wait, q6 is not adj q1 (v). So x6 not adj v.
                            # Also q6 not adj q0 (u). So x6 not adj u.
                            
                            cand6 = []
                            for w in range(n):
                                if w in used:
                                    continue
                                if not adj[x2][w] or not adj[x4][w]:
                                    continue
                                if adj[u][w] or adj[v][w]:
                                    continue
                                cand6.append(w)
                            
                            cand7 = []
                            for w in range(n):
                                if w in used:
                                    continue
                                if not adj[x3][w] or not adj[x5][w]:
                                    continue
                                if adj[u][w] or adj[v][w]:
                                    continue
                                cand7.append(w)
                            
                            for x6 in cand6:
                                for x7 in cand7:
                                    if x6 == x7:
                                        continue
                                    # Check x6 adj x7
                                    if not adj[x6][x7]:
                                        continue
                                    # Check remaining edges
                                    # q2(0's nbr) adj q6: already checked
                                    # q4 adj q6: already checked
                                    # q3 adj q7: already checked
                                    # q5 adj q7: already checked
                                    # q6 adj q7: checked
                                    # Also need:
                                    # q2 not adj q3, q5, q7 (in Q3, 2 is adj to 0,4,6 only)
                                    # q4 not adj q3, q5, q7 (in Q3, 4 is adj to 0,2,6 only)
                                    # q3 not adj q2, q4, q6 (in Q3, 3 is adj to 1,5,7 only)
                                    # q5 not adj q2, q4, q6 (in Q3, 5 is adj to 1,3,7 only)
                                    # q6 not adj q3, q5 (in Q3, 6 is adj to 2,4,7 only)
                                    # q7 not adj q2, q4 (in Q3, 7 is adj to 3,5,6 only)
                                    
                                    if adj[x2][x3] or adj[x2][x5] or adj[x2][x7]:
                                        continue
                                    if adj[x4][x3] or adj[x4][x5] or adj[x4][x7]:
                                        continue
                                    if adj[x3][x2] or adj[x3][x4] or adj[x3][x6]:
                                        continue
                                    if adj[x5][x2] or adj[x5][x4] or adj[x5][x6]:
                                        continue
                                    if adj[x6][x3] or adj[x6][x5]:
                                        continue
                                    if adj[x7][x2] or adj[x7][x4]:
                                        continue
                                    
                                    return True
            return False
        
        if not has_q3_with_edge(adj, u, v):
            valid_additions.append(ce)
        adj[u][v] = 0
        adj[v][u] = 0

    # Add valid edges one by one (greedy)
    # We'll try to add as many as possible
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for ce in list(valid_additions):
            u, v = ce
            if adj[u][v] == 1:
                continue
            adj[u][v] = 1
            adj[v][u] = 1
            # Recheck: adding this edge might create a Q3 with other previously added edges
            # We need to check if the graph now has a Q3
            # This is the expensive full check
            # But we can use the targeted check again: check if there's a Q3 using edge (u,v)
            if has_q3_with_edge(adj, u, v):
                adj[u][v] = 0
                adj[v][u] = 0
                valid_additions.remove(ce)
            else:
                edges.append(ce)
                valid_additions.remove(ce)
                changed = True
                best_edges = list(edges)
                best_count = len(edges)
                # Update candidate_edges: remove any edge sharing u or v? No, just mark as used.
                break  # Restart the loop to re-evaluate
    
    return best_edges

