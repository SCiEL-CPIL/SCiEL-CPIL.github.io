import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # 1. Generate Paley graph on 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a set of neighbors of i
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute K4s and I4s (independent 4-sets)
    # K4: 4 vertices, all connected
    # I4: 4 vertices, no connections (clique in complement)
    
    def find_k4s(n, adj):
        k4s = []
        # Iterate over all 4-subsets? 37 choose 4 is ~ 66k. Very fast.
        for a in range(n):
            for b in range(a+1, n):
                if b not in adj[a]: continue
                for c in range(b+1, n):
                    if c not in adj[a] or c not in adj[b]: continue
                    for d in range(c+1, n):
                        if d in adj[a] and d in adj[b] and d in adj[c]:
                            k4s.append((a, b, c, d))
        return k4s

    def find_i4s(n, adj):
        i4s = []
        # Complement adjacency
        # a, b, c, d form I4 if no edges between them
        for a in range(n):
            for b in range(a+1, n):
                if b in adj[a]: continue
                for c in range(b+1, n):
                    if c in adj[a] or c in adj[b]: continue
                    for d in range(c+1, n):
                        if d not in adj[a] and d not in adj[b] and d not in adj[c]:
                            i4s.append((a, b, c, d))
        return i4s

    k4_list = find_k4s(n, adj)
    i4_list = find_i4s(n, adj)
    
    def try_add_vertex(n, adj, k4_list, i4_list):
        # Find S subset of {0..n-1} such that:
        # 1. S contains no K4
        # 2. S intersects every I4 (so V\S contains no I4)
        
        # Backtracking search for S
        # Variables: 0..n-1. State: 0 (not in S), 1 (in S), -1 (undecided)
        # Constraints:
        # - For each K4 in k4_list: not (all 4 in S)
        # - For each I4 in i4_list: at least one in S
        
        # To make it faster, we can use a greedy randomized approach first, then repair.
        # Or simple backtracking with MRV (Minimum Remaining Values) heuristic.
        
        # Given the small size, let's try a simple recursive backtracking.
        # We need to be careful with recursion depth and speed.
        
        # Alternative: Use a SAT solver? No external libs.
        # Let's try a greedy construction of S.
        # Start with S = empty.
        # We must hit all I4s.
        # We must avoid containing any K4.
        
        # This is equivalent to: Find a set S that is a vertex cover for the hypergraph of I4s, 
        # and an independent set for the hypergraph of K4s.
        
        # Let's use a simple local search.
        # Start with S = random subset.
        # Score = (# K4s contained in S) + (# I4s disjoint from S)
        # Minimize Score. If Score == 0, we have a valid S.
        
        best_score = float('inf')
        best_s = None
        
        for _ in range(500): # 500 restarts
            s = [random.choice([0, 1]) for _ in range(n)]
            # Local search
            for step in range(1000):
                # Calculate score
                score = 0
                # Check K4s
                for k4 in k4_list:
                    if all(s[v] == 1 for v in k4):
                        score += 1
                # Check I4s
                for i4 in i4_list:
                    if all(s[v] == 0 for v in i4):
                        score += 1
                        
                if score == 0:
                    return [i for i in range(n) if s[i] == 1]
                
                # Find a bit to flip
                # Pick a random violation
                # If a K4 is fully in S, flip one of its bits to 0
                # If an I4 is fully out of S, flip one of its bits to 1
                
                flipped = False
                for k4 in k4_list:
                    if all(s[v] == 1 for v in k4):
                        v = random.choice(k4)
                        s[v] = 0
                        flipped = True
                        break
                if not flipped:
                    for i4 in i4_list:
                        if all(s[v] == 0 for v in i4):
                            v = random.choice(i4)
                            s[v] = 1
                            flipped = True
                            break
                if not flipped:
                    # No violations? Should have returned.
                    # Or maybe score > 0 but no single flip fixes it?
                    # Random flip
                    v = random.randrange(n)
                    s[v] = 1 - s[v]
                    
                if time.time() > deadline:
                    return None
                    
        return None

    # Try to add vertices
    while time.time() < deadline:
        S = try_add_vertex(n, adj, k4_list, i4_list)
        if S is None:
            break
            
        # Add new vertex n
        new_v = n
        adj.append(set(S))
        for v in S:
            adj[v].add(new_v)
            
        # Update k4_list and i4_list
        # New K4s: {new_v} U T where T is a K3 in S
        # Find all K3s in S
        new_k4s = []
        for a in S:
            for b in S:
                if b <= a: continue
                if b not in adj[a]: continue
                for c in S:
                    if c <= b: continue
                    if c not in adj[a] or c not in adj[b]: continue
                    new_k4s.append((new_v, a, b, c))
        k4_list.extend(new_k4s)
        
        # New I4s: {new_v} U T where T is an I3 in V \ S
        # V \ S are the non-neighbors
        non_s = [v for v in range(n) if v not in S]
        new_i4s = []
        for a in non_s:
            for b in non_s:
                if b <= a: continue
                if b in adj[a]: continue
                for c in non_s:
                    if c <= b: continue
                    if c in adj[a] or c in adj[b]: continue
                    new_i4s.append((new_v, a, b, c))
        i4_list.extend(new_i4s)
        
        n += 1
        
    # Return the graph
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges
