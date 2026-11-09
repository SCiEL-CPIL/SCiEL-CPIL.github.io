import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency list as bitmasks for efficiency
    # adj[i] is an integer bitmask where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges

    def find_k4s(adj, n):
        k4s = []
        # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
        # We can iterate through combinations of 4 vertices.
        # n is small (37-42), so C(42,4) is ~100k, which is fast.
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)) and \
               (adj[b] & (1 << c)) and (adj[b] & (1 << d)) and \
               (adj[c] & (1 << d)):
                k4s.append(combo)
        return k4s

    def find_ind4s(adj, n):
        ind4s = []
        # An independent set of 4 is a set of 4 vertices with no edges between them.
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if not (adj[a] & (1 << b)) and not (adj[a] & (1 << c)) and not (adj[a] & (1 << d)) and \
               not (adj[b] & (1 << c)) and not (adj[b] & (1 << d)) and \
               not (adj[c] & (1 << d)):
                ind4s.append(combo)
        return ind4s

    def try_add_vertex(adj, n, k4s, ind4s):
        # We want to find S subset of V(n) such that:
        # 1. No K4 is a subset of S.
        # 2. No Ind4 is disjoint from S (i.e., intersects S).
        
        # Randomized local search
        best_score = float('inf')
        best_S = None
        
        # Precompute masks for K4s and Ind4s to check quickly?
        # Actually, checking constraints for a candidate S:
        # Violation 1: Any K4 C such that C subset S.
        # Violation 2: Any Ind4 I such that I intersect S is empty (i.e., I subset V\S).
        
        def score(S):
            v1 = 0
            for C in k4s:
                # Check if C is subset of S
                mask_C = 0
                for v in C:
                    mask_C |= (1 << v)
                if (S & mask_C) == mask_C:
                    v1 += 1
            v2 = 0
            for I in ind4s:
                mask_I = 0
                for v in I:
                    mask_I |= (1 << v)
                if (S & mask_I) == 0:
                    v2 += 1
            return v1 + v2

        # To make it faster, precompute masks for k4s and ind4s
        k4_masks = []
        for C in k4s:
            m = 0
            for v in C:
                m |= (1 << v)
            k4_masks.append(m)
        
        ind4_masks = []
        for I in ind4s:
            m = 0
            for v in I:
                m |= (1 << v)
            ind4_masks.append(m)
            
        def fast_score(S):
            v1 = 0
            for m in k4_masks:
                if (S & m) == m:
                    v1 += 1
            v2 = 0
            for m in ind4_masks:
                if (S & m) == 0:
                    v2 += 1
            return v1 + v2

        # Local search
        S = 0
        # Start with random S
        for _ in range(1000):
            S = random.getrandbits(n)
            # Simple hill climbing
            for step in range(1000):
                if time.time() > deadline:
                    return None
                curr_score = fast_score(S)
                if curr_score == 0:
                    return S
                # Try flipping a random bit
                v = random.randint(0, n - 1)
                S_new = S ^ (1 << v)
                new_score = fast_score(S_new)
                if new_score < curr_score:
                    S = S_new
                elif new_score == curr_score and random.random() < 0.1:
                    S = S_new
        return None

    # Iteratively add vertices
    for target_n in range(n + 1, 47): # Try up to 46
        if time.time() > deadline - 10:
            break
        
        # Update K4s and Ind4s
        k4s = find_k4s(adj, n)
        ind4s = find_ind4s(adj, n)
        
        # Try to add a vertex
        added = False
        attempts = 0
        while attempts < 20 and not added and time.time() < deadline - 10:
            attempts += 1
            S = try_add_vertex(adj, n, k4s, ind4s)
            if S is not None:
                # Add new vertex n
                new_adj = adj + [0]
                # Connect new vertex to bits set in S
                for v in range(n):
                    if S & (1 << v):
                        new_adj[n] |= (1 << v)
                        new_adj[v] |= (1 << n)
                adj = new_adj
                n += 1
                added = True
                # Reset attempts for next vertex
                attempts = 0
                
        if not added:
            break
            
    # Return the best graph found
    edges = get_edges(adj, n)
    return n, edges

# EVOLVE-BLOCK-END
