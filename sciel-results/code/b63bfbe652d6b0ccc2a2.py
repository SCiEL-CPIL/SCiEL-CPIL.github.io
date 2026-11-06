import time
import random
import itertools

def _has_c5(n, adj):
    # Check for C5 subgraph
    # A C5 is a cycle of length 5.
    # We can check for paths of length 4 that close.
    # Or simpler: iterate all 5-subsets? 9 choose 5 = 126. Fast.
    # For each subset, check if it contains a C5.
    # A graph on 5 vertices contains a C5 if it is isomorphic to C5 or has C5 as subgraph.
    # Actually, if the induced subgraph on 5 vertices contains a C5, we are done.
    # But C5 can be non-induced.
    # However, if a graph contains C5, it contains a cycle of length 5.
    # We can just check if the subgraph induced by any 5 vertices contains a cycle of length 5.
    # Or simply: does the graph contain a cycle of length 5?
    
    # Efficient check for C5 in small graph:
    # Try all starting nodes u.
    # BFS/DFS for paths of length 4?
    # Since n is small (<=11), we can use networkx if available, but std lib only.
    # Let's use a simple check:
    # A C5 exists if there is a sequence u1, u2, u3, u4, u5 such that
    # (u1,u2), (u2,u3), (u3,u4), (u4,u5), (u5,u1) are edges.
    
    # We can iterate permutations? Too slow.
    # Iterate u1, u2, u3, u4. Check if u3 and u4 have a common neighbor u5 != u1, u2, u3, u4 such that u5 connected to u1?
    
    # Simpler: 
    # For each pair (u, v) non-adjacent? No.
    
    # Let's just check all 5-cycles explicitly.
    # n <= 11.
    # Pick 5 distinct vertices. Check if they contain a C5.
    # A set of 5 vertices contains a C5 if the induced subgraph (or subgraph) has a cycle of length 5.
    # Since we only care about existence of C5 in the whole graph, we can check if any 5-vertex subset induces a graph that contains C5.
    # Actually, if G contains C5, then some 5-vertex subset induces a graph that contains C5.
    
    for combo in itertools.combinations(range(n), 5):
        # Check if the subgraph induced by combo contains a C5
        # Map combo to 0..4
        indices = {v: i for i, v in enumerate(combo)}
        # Build local adj matrix 5x5
        local_adj = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j and adj[combo[i]][combo[j]]:
                    local_adj[i][j] = 1
        
        # Check if local_adj contains C5
        # A 5-vertex graph contains C5 iff it is isomorphic to C5, or has C5 as subgraph.
        # Since it has 5 vertices, if it contains C5, it MUST be C5 (possibly with extra chords).
        # So we just need to check if there is a cycle of length 5.
        # In a 5-vertex graph, a cycle of length 5 uses all vertices.
        # So we check if there is a permutation (0,1,2,3,4) such that edges exist.
        # There are 5! / (2*5) = 12 unique cycles.
        # We can just check if the graph is not C5-free.
        # C5-free graphs on 5 vertices:
        # - Tree (0 cycles)
        # - Unicyclic with cycle len != 5 (impossible for 5 verts, min cycle 3)
        # - Contains C3, C4.
        # So if it doesn't contain C5, it must have girth < 5 or be acyclic.
        
        # Let's just check for the specific cycle pattern.
        # We can check if the complement has a matching? No.
        
        # Simple check: 
        # Does local_adj contain a cycle of length 5?
        # We can try all 12 cyclic permutations of 0,1,2,3,4.
        # If any forms a cycle, return True.
        found = False
        # Standard C5: 0-1-2-3-4-0
        # We can just check if the graph is isomorphic to a supergraph of C5.
        # This is equivalent to: Can we find a bijection f: {0..4} -> {0..4} such that (f(i), f(i+1)) is an edge?
        # Since it's small, let's just hardcode the 12 cycles or use a simple check.
        
        # Check if there is a cycle of length 5.
        # We can use a simple DFS or just check the 12 permutations.
        # Permutations of (1,2,3,4) starting at 0.
        # 0-1-2-3-4-0
        # 0-1-2-4-3-0
        # 0-1-3-2-4-0
        # 0-1-3-4-2-0
        # 0-1-4-2-3-0
        # 0-1-4-3-2-0
        # (and reversals, but undirected so same)
        
        perms = [
            [0,1,2,3,4],
            [0,1,2,4,3],
            [0,1,3,2,4],
            [0,1,3,4,2],
            [0,1,4,2,3],
            [0,1,4,3,2]
        ]
        for p in perms:
            if local_adj[p[0]][p[1]] and local_adj[p[1]][p[2]] and local_adj[p[2]][p[3]] and local_adj[p[3]][p[4]] and local_adj[p[4]][p[0]]:
                return True
    return False

def _alpha(n, adj):
    # Max independent set size
    # For n <= 11, we can use bitmask DP or just check all subsets?
    # 2^11 = 2048. Very fast.
    max_alpha = 0
    for mask in range(1 << n):
        # Check if mask is independent
        is_ind = True
        for i in range(n):
            if mask & (1 << i):
                for j in range(i+1, n):
                    if mask & (1 << j):
                        if adj[i][j]:
                            is_ind = False
                            break
            if not is_ind:
                break
        if is_ind:
            cnt = bin(mask).count("1")
            if cnt > max_alpha:
                max_alpha = cnt
    return max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Default best solution: 8 K4s
    best_n = 32
    best_edges = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Try to find a better block H (n_H, alpha_H) to replace some K4s
    # We want to maximize n_total = n_H + k * 4
    # subject to alpha_H + k * 1 <= 8
    # and n_total <= 35
    
    # Possible k values:
    # If alpha_H = 1, k=7, n_total = n_H + 28. Max n_H for alpha=1 is 4. n=32.
    # If alpha_H = 2, k=6, n_total = n_H + 24. We want n_H > 8.
    # If alpha_H = 3, k=5, n_total = n_H + 20. We want n_H > 13.
    # If alpha_H = 4, k=4, n_total = n_H + 16. We want n_H > 19.
    
    # Search for H with n_H in [9, 11] and alpha_H <= 2.
    # If found, n_total = n_H + 24.
    # If n_H=9, n_total=33.
    # If n_H=10, n_total=34.
    # If n_H=11, n_total=35.
    
    # We will search for a C5-free graph on n_H vertices with alpha <= 2.
    
    def find_block(target_n, max_alpha):
        # Randomized search
        # Start with random graph, try to remove edges to kill C5, add edges to kill alpha?
        # Better: Start with complete graph, remove edges to kill C5?
        # Complete graph has alpha=1. It contains C5.
        # We need to remove enough edges to kill all C5s, while keeping alpha <= max_alpha.
        
        # Let's try a heuristic:
        # Generate random graphs, check constraints.
        
        # Or: Start with empty graph (alpha=n). Add edges to reduce alpha.
        # This is hard.
        
        # Let's try: Start with a random graph.
        # Check C5. If C5, remove random edge from cycle.
        # Check Alpha. If Alpha > max_alpha, add random edge to reduce alpha.
        
        adj = [[0]*target_n for _ in range(target_n)]
        
        # Initialize with random edges
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        for step in range(1000):
            if time.time() > deadline:
                return None
            
            # Check C5
            if _has_c5(target_n, adj):
                # Find a C5 and remove an edge
                # We need to find the cycle.
                # Re-use the logic from _has_c5 but return the cycle
                # For speed, just pick a random edge and remove it?
                # Better: Find a C5.
                # We'll just remove a random edge and hope it helps.
                # Or iterate edges and check if removing it kills a C5? Too slow.
                
                # Let's just pick a random edge and remove it.
                # This might take many steps.
                
                # Alternative: Use the _has_c5 logic to find one.
                # Let's modify _has_c5 to return the cycle?
                # For now, let's just try random removal.
                
                i, j = random.sample(range(target_n), 2)
                if adj[i][j]:
                    adj[i][j] = 0
                    adj[j][i] = 0
                else:
                    # Add an edge?
                    # If alpha is too high, we should add edges.
                    # Let's check alpha first.
                    pass
            
            # Check Alpha
            a = _alpha(target_n, adj)
            if a > max_alpha:
                # Add an edge to reduce alpha
                # Pick two non-adjacent vertices that are in a max independent set?
                # Just pick random non-adjacent pair.
                added = False
                for _ in range(10):
                    i, j = random.sample(range(target_n), 2)
                    if not adj[i][j]:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        added = True
                        break
                if not added:
                    return None # Complete graph, alpha=1, but has C5. Stuck.
            
            # Check if valid
            if not _has_c5(target_n, adj) and _alpha(target_n, adj) <= max_alpha:
                # Convert to edges
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                return edges
        return None

    # Try to find H for n_H = 11, 10, 9
    # Priority: Higher n_H gives higher total n.
    
    for n_h in [11, 10, 9]:
        # We need alpha_h <= 2 for k=6 (n_total = n_h + 24)
        # If n_h=11, alpha_h must be <= 2?
        # 11 + 24 = 35.
        # If we find H(11, alpha<=2), we get 35.
        
        edges_h = find_block(n_h, 2)
        if edges_h:
            # Construct total graph
            total_edges = []
            # Add H
            for e in edges_h:
                total_edges.append([e[0], e[1]])
            
            # Add 6 K4s
            base = n_h
            for block in range(6):
                b = base + block * 4
                for i in range(4):
                    for j in range(i + 1, 4):
                        total_edges.append([b + i, b + j])
            
            total_n = n_h + 24
            if total_n > best_n:
                best_n = total_n
                best_edges = total_edges
            
            # Update deadline check
            if time.time() > deadline:
                break
            
    return {'n': best_n, 'edges': best_edges}
