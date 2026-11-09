import time
import random

def has_triangle(n, adj):
    # adj is a list of sets or bitmasks
    # For speed, use bitmasks
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                # Check common neighbors
                if adj[u] & adj[v]:
                    return True
    return False

def has_independent_set_10(n, adj):
    # adj[u] is a bitmask of neighbors of u
    # We want to find a set of 10 vertices with no edges between them.
    # This is equivalent to finding a clique of size 10 in the complement graph.
    # Or just DFS on the original graph avoiding edges.
    
    # Greedy upper bound check first?
    # If max degree is high, independent set is small.
    # But let's do a DFS with pruning.
    
    # Order vertices by degree (descending) to prune faster?
    # Actually, for independent set, we want to pick vertices that have few connections to remaining candidates.
    
    # Simple DFS
    def dfs(candidates, count):
        if count == 10:
            return True
        if not candidates:
            return False
        # Heuristic: pick a vertex from candidates
        # To speed up, if the number of candidates is small, check directly
        # If candidates size < 10 - count, return False
        if len(candidates) < 10 - count:
            return False
            
        # Pick a vertex, say the first one
        # Better: pick a vertex that reduces candidates the most?
        # For now, just pick the first.
        u = candidates[0]
        # Branch 1: Include u
        # New candidates are those in candidates[1:] that are NOT neighbors of u
        new_cands = []
        for v in candidates[1:]:
            if not (adj[u] & (1 << v)):
                new_cands.append(v)
        if dfs(new_cands, count + 1):
            return True
        # Branch 2: Exclude u
        if dfs(candidates[1:], count):
            return True
        return False

    # Initial candidates: all vertices
    # Optimization: If the graph is dense, this is fast.
    # If sparse, it might be slow.
    # But triangle-free graphs with alpha < 10 are not too sparse.
    
    # To handle potential timeout in DFS, we add a node limit.
    # But for n=40, it should be okay.
    
    return dfs(list(range(n)), 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Try to find a valid graph for n=40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Generate candidate graphs
        candidates = []
        
        # Candidate 1: Andrasfai-like
        # S = {s in 1..n//2 | s % 3 == 1}
        S1 = [s for s in range(1, n // 2 + 1) if s % 3 == 1]
        if S1:
            candidates.append(S1)
            
        # Candidate 2: S = {s in 1..n//2 | s % 3 == 2}
        S2 = [s for s in range(1, n // 2 + 1) if s % 3 == 2]
        if S2:
            candidates.append(S2)
            
        # Candidate 3: Quadratic Residues (if n is prime)
        # For simplicity, skip for now.
        
        # Candidate 4: Random sets of size approx n/3
        # We need them to be triangle-free.
        # We can generate random sets and check for triangles, keeping only valid ones.
        # But checking triangles for every random set is expensive.
        # Instead, we can just try a few random sets and check validity.
        
        for _ in range(10):
            if time.time() > deadline:
                break
            # Random set of size k
            k = n // 3
            # Ensure symmetry: if s in S, n-s in S
            # Pick k/2 values from 1..n//2
            half_n = n // 2
            if k % 2 == 0:
                size_half = k // 2
            else:
                size_half = k // 2 + 1
            if size_half > half_n:
                size_half = half_n
            random_half = random.sample(range(1, half_n + 1), size_half)
            S_rand = random_half + [n - x for x in random_half if n - x != x]
            # Remove duplicates and sort
            S_rand = sorted(list(set(S_rand)))
            if S_rand not in candidates:
                candidates.append(S_rand)
                
        for S in candidates:
            if time.time() > deadline:
                break
            
            # Build adjacency bitmask
            adj = [0] * n
            for u in range(n):
                for s in S:
                    v1 = (u + s) % n
                    v2 = (u - s) % n
                    if v1 != u:
                        adj[u] |= (1 << v1)
                    if v2 != u:
                        adj[u] |= (1 << v2)
            
            # Check triangle-free
            if has_triangle(n, adj):
                continue
                
            # Check independent set < 10
            if not has_independent_set_10(n, adj):
                # Valid graph found
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    # Fallback: Return the seed graph
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
