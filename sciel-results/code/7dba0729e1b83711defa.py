N = 40

# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 55  # 55s safety margin
    
    # Base: cycle C40 with rim edges
    rim_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Add chords from vertex 0 to even vertices 2, 4, ..., 38
    chords = [(0, 2 * j) for j in range(1, N // 2)]
    
    all_edges = rim_edges + chords
    edge_set = set()
    for u, v in all_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency
    adj = {i: set() for i in range(N)}
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_cycle_of_length(L):
        """Find a simple cycle of exactly L vertices using DFS with memoization."""
        if L == N:
            # Full rim cycle
            return list(range(N))
        
        # Try using vertex 0 in the cycle with chords
        # Strategy: use chord (0, 2k) to create a shorter cycle
        # For cycle of length L, we need L-1 edges in a path from 0 back to 0
        
        # Approach: find a Hamiltonian path of length L-1 in the induced subgraph
        # that starts at 0, ends at a neighbor of 0, with all L vertices distinct
        
        # For small L, brute force with backtracking
        if L <= 20:
            def dfs(path, visited):
                if time.time() > deadline:
                    return None
                if len(path) == L:
                    # Check if last vertex connects back to first
                    if path[-1] in adj[path[0]]:
                        return path[:]
                    return None
                last = path[-1]
                for nb in sorted(adj[last]):
                    if nb not in visited:
                        visited.add(nb)
                        result = dfs(path + [nb], visited)
                        if result:
                            return result
                        visited.remove(nb)
                return None
            
            result = dfs([0], {0})
            if result:
                return result
        
        # For larger L, use the structure: rim + chords from 0
        # A cycle of length L can be formed by:
        # - Using the rim from a to b (going forward)
        # - Plus a chord from 0 to some vertex
        # - Plus rim segments
        
        # Key insight: with chords (0, 2j), we can form cycles of many lengths
        # Cycle: 0 - 2j - (2j+1) - ... - (2j+L-2) - (2j+L-1) - 0
        # This requires edge (2j+L-1, 0) to exist
        # (2j+L-1, 0) is a chord if 2j+L-1 is even, i.e., L-1 is even, i.e., L is odd
        # Or (2j+L-1, 0) is a rim edge if 2j+L-1 == 39 or 2j+L-1 == 1
        
        # Alternative: 0 - 2j - (2j-1) - ... - (2j-L+2) - (2j-L+1) - 0
        # Going backward on rim, then need (2j-L+1, 0) edge
        
        # Let me try: for any L from 3 to 40, construct explicitly
        # Use vertex 0 and two chord neighbors 2a and 2b
        # Path: 0 - 2a - (2a+1) - ... - (2b-1) - 2b - 0 (going forward on rim)
        # Length: 1 (0 to 2a) + (2b - 2a - 1) rim edges + 1 (2b to 0) = 2b - 2a + 1 vertices? No
        # Vertices: 0, 2a, 2a+1, ..., 2b, that's 2b - 2a + 1 vertices including 0? 
        # 0, then 2a, 2a+1, ..., 2b: that's (2b - 2a + 1) rim vertices + 0 = 2b - 2a + 2 vertices
        # Edges: (0,2a), (2a,2a+1), ..., (2b-1,2b), (2b,0): that's 1 + (2b-2a-1) + 1 = 2b-2a+1 edges = 2b-2a+2? 
        # No: number of edges = number of vertices for a cycle
        # Vertices: 0, 2a, 2a+1, ..., 2b. Count: 1 + (2b - 2a + 1) = 2b - 2a + 2
        # Edges: (0,2a), (2a,2a+1), ..., (2b-1,2b), (2b,0). Count: 1 + (2b - 2a) = 2b - 2a + 1
        # Mismatch. Let me recount.
        
        # Path: 0 -> 2a -> 2a+1 -> ... -> 2b -> 0
        # Vertices in order: [0, 2a, 2a+1, 2a+2, ..., 2b]
        # Number of vertices: 1 + (2b - 2a + 1) = 2b - 2a + 2
        # Edges: (0,2a), (2a,2a+1), (2a+1,2a+2), ..., (2b-1,2b), (2b,0)
        # Number of edges: 1 + (2b - 2a) + 1 = 2b - 2a + 2
        # Good, edges = vertices, so it's a cycle of length 2b - 2a + 2
        
        # So we can form cycles of length L = 2b - 2a + 2 where 1 <= a < b <= 19
        # L ranges from 2(2-1)+2=4 to 2(19-1)+2=38
        # So even lengths 4 through 38 are covered.
        
        # For odd lengths and length 3, 39, 40:
        # L=3: 0-2-4-0? Edges (0,2), (2,4)? No, (2,4) is not an edge. 0-2-3-0? (3,0) not an edge.
        # Hmm, need to check. 0-2-1-0? (1,0) not an edge. 
        # Wait, rim edges are (i, i+1 mod 40). So (1,0) is NOT a rim edge (rim has (0,1) and (39,0)).
        # (0,1) IS a rim edge. So 0-1-2-0? Edges (0,1) rim, (1,2) rim, (2,0) chord. Yes! L=3.
        
        # L=39: 0-2-3-4-...-39-0? Vertices: 0,2,3,...,39 = 39 vertices. 
        # Edges: (0,2) chord, (2,3)...(38,39) rim, (39,0) rim. Count: 1+37+1=39. Yes!
        
        # L=40: full rim.
        
        # For odd L >= 5: 
        # Try: 0-2a-(2a-1)-...-2b-0 going backward
        # Or: use the rim in a different way
        # L=5: 0-2-3-4-5-0? (5,0) not an edge. 0-2-1-39-38-0? (2,1) rim, (1,39)? No.
        # 0-2-3-1-0? (3,1) not an edge.
        # 0-1-2-4-0? (1,2) rim, (2,4)? No.
        # Hmm, let me just do BFS/DFS for each length.
        
        # Actually, let me just verify all cycles exist by searching
        return None
    
    # Generate cycles for all lengths 3 to 40
    cycles = []
    for L in range(3, N + 1):
        if time.time() > deadline:
            # Fallback to seed if running out of time
            break
        
        found = None
        
        if L == N:
            found = list(range(N))
        elif L == 3:
            # 0-1-2-0: edges (0,1) rim, (1,2) rim, (2,0) chord
            found = [0, 1, 2]
        elif L == 39:
            # 0-2-3-...-39-0
            found = [0] + list(range(2, N))
        else:
            # For even L = 2m where 4 <= L <= 38:
            # L = 2b - 2a + 2, so b - a = (L-2)/2
            # a >= 1, b <= 19
            # If L is even:
            if L % 2 == 0:
                diff = (L - 2) // 2
                # Need a >= 1, a + diff <= 19
                a = 1
                b = a + diff
                if b <= 19:
                    # Cycle: 0, 2a, 2a+1, ..., 2b
                    found = [0] + list(range(2 * a, 2 * b + 1))
                else:
                    # Try different a
                    a = 19 - diff
                    b = 19
                    if a >= 1:
                        found = [0] + list(range(2 * a, 2 * b + 1))
            # For odd L (5, 7, ..., 37):
            # Need a different construction
            # Try: 0 - 1 - (L-2) - (L-3) - ... - 2 - 0? 
            # Or: 0 - 2 - 3 - ... - (L-1) - (L-2) - 0? No.
            # Let's use: 0 - 2a - (2a+1) - ... - (2a+L-2) - (2a+L-3) - 0
            # This needs edge (2a+L-3, 0)
            # Or going the other way on the rim segment
            # 
            # Alternative: 0 - 2a - (2a-1) - (2a-2) - ... - 2b - 0
            # Vertices: 0, 2a, 2a-1, ..., 2b. Count: 1 + (2a - 2b + 1) = 2a - 2b + 2
            # So L = 2a - 2b + 2, a > b, a,b in [1,19]
            # Same as before, just going backward. Still even lengths only.
            
            # For odd lengths, try: include vertex 1
            # 0 - 1 - 2 - 3 - ... - (L-1) - 0? Need (L-1, 0) edge
            # (L-1, 0) is chord if L-1 is even, i.e., L is odd. 
            # But L-1 must be in {2,4,...,38}
            # So L-1 = 2j, L = 2j+1, j in [1,19], L in [3,39]
            # Cycle: 0, 1, 2, ..., 2j. Count: 2j+1 = L. 
            # Edges: (0,1) rim, (1,2) rim, ..., (2j-1,2j) rim, (2j,0) chord.
            # This works for odd L = 3, 5, 7, ..., 39!
            # But we already handled L=3 and L=39.
            # For odd L in [5, 37]:
            j = (L - 1) // 2
            if 1 <= j <= 19:
                found = list(range(0, 2 * j + 1))
        
        if found is None or len(found) != L or len(set(found)) != L:
            # Fallback: DFS search
            def dfs_search(target_len):
                if time.time() > deadline:
                    return None
                def dfs(path, visited):
                    if time.time() > deadline:
                        return None
                    if len(path) == target_len:
                        if path[-1] in adj[path[0]]:
                            return path[:]
                        return None
                    last = path[-1]
                    for nb in sorted(adj[last]):
                        if nb not in visited:
                            visited.add(nb)
                            result = dfs(path + [nb], visited)
                            if result:
                                return result
                            visited.remove(nb)
                    return None
                return dfs([0], {0})
            
            found = dfs_search(L)
        
        if found is None or len(found) != L:
            # Last resort: use seed cycles (which are valid but may not use minimal edges)
            # Actually, we must have valid cycles. Let's just use the rim-based approach
            # and hope our construction is correct. If not, fall back to a known good cycle.
            # For safety, use a cycle that definitely exists in our graph
            if L <= N:
                # Use vertices 0, 1, 2, ..., L-1 with appropriate edges
                # This may not work, so let's just assert we found it
                raise ValueError(f"Failed to find cycle of length {L}")
        
        cycles.append(found)
    
    # Verify all cycles are valid
    for i, cyc in enumerate(cycles):
        L = i + 3
        assert len(cyc) == L, f"Cycle {L}: expected {L} vertices, got {len(cyc)}"
        assert len(set(cyc)) == L, f"Cycle {L}: not all distinct"
        for j in range(L):
            u = cyc[j]
            v = cyc[(j + 1) % L]
            assert (min(u, v), max(u, v)) in edge_set, f"Cycle {L}: edge ({u},{v}) missing"
    
    return {'edges': list(edge_set), 'cycles': cycles}
# EVOLVE-BLOCK-END
