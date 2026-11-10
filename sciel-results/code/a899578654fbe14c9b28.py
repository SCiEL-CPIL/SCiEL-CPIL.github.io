import itertools
import random
import time
import math

def construct():
    n = 18
    
    # Precompute all 3-bitmasks (triangles) for all possible triples?
    # No, triangles depend on the graph.
    # But we can precompute the list of all possible triples as masks.
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((1 << i) | (1 << j) | (1 << k))
                
    def is_k4_free(adj):
        # Check all 4-tuples
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if (adj[k] & common) != 0:
                        return False
        return True

    def get_triangles(adj):
        # Returns list of masks that form a triangle in adj
        tris = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j: # Avoid duplicates, though mask handles it
                         tris.append((1 << i) | (1 << j) | (1 << k))
                    elif k < i:
                         # This case shouldn't happen if we iterate i<j and k from common
                         # common contains neighbors of i and j.
                         # If k < i, it's still a valid triangle.
                         # But we want unique masks.
                         pass
        # The above logic might miss some or double count if not careful.
        # Let's just iterate all triples and check.
        tris = []
        for mask in all_triples:
            # Extract vertices
            # Check if all 3 edges exist
            # This is O(1) if we have adj
            # But we can do it faster:
            # Let vertices be v1, v2, v3
            # We can precompute the vertices for each mask?
            pass
        
        # Faster: Iterate i<j<k
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1: continue
                common = adj[i] & adj[j]
                # Neighbors of i and j
                # For each k in common, if k > j, we have a triangle i,j,k
                # If k < i, we have triangle k,i,j
                # If i < k < j, we have triangle i,k,j
                # To avoid duplicates, we can just check if edge (i,j), (j,k), (i,k) exist.
                # Let's just check the 3 edges.
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    # Triangle i, j, k exists if edges (i,j), (j,k), (i,k) exist.
                    # We know (i,j) exists.
                    # k is in common, so (i,k) and (j,k) exist.
                    # So yes, it's a triangle.
                    tris.append((1 << i) | (1 << j) | (1 << k))
        return tris

    def get_score(adj, tris):
        # Find max s such that some subset of size s is triangle-free
        # Triangle-free means it does NOT contain any triangle mask from tris
        # A subset mask M contains a triangle T if (M & T) == T
        
        # We can iterate s from n down to 1
        # For each s, check all subsets of size s
        # Optimization: If tris is empty, return n
        if not tris:
            return n
            
        for s in range(n, 0, -1):
            # Generate all masks of size s
            # Using combinations is slow.
            # We can use a recursive generator or bit tricks.
            # For n=18, s around 10-12, combinations are ~18k.
            # 18k * len(tris) checks.
            # If len(tris) is ~100, that's 1.8M ops per s.
            # 18 s values -> 30M ops. Too slow for every step in SA.
            
            # Better: Just check if ANY subset of size s is triangle-free.
            # This is a "does there exist a set of size s with no triangle" problem.
            # This is equivalent to: The independence number of the hypergraph of triangles?
            # No.
            
            # Let's stick to the combination check but optimize it.
            # Precompute list of masks for each size?
            # There are 2^18 masks. We can precompute popcount.
            
            # Actually, we can just iterate all masks and check popcount?
            # 2^18 = 262k masks.
            # For each mask, check if it contains any triangle.
            # If yes, it's not a candidate for "largest triangle-free".
            # We want the max popcount of a mask that contains NO triangle.
            
            # This can be done in one pass over all 2^18 masks!
            # For each mask M:
            #   if M contains any triangle: skip
            #   else: update max_size = max(max_size, popcount(M))
            
            # How to check "M contains any triangle" fast?
            # Iterate tris: if (M & t) == t: break
            # Total ops: 2^18 * len(tris).
            # If len(tris) is 100, 26M ops.
            # This is ~0.05s in C, but ~5s in Python.
            # Too slow for every SA step.
            
            # We need a faster score function.
            # Approximation?
            # Or, we can cache? No, graph changes.
            
            # Alternative:
            # The score is the size of the largest triangle-free induced subgraph.
            # This is the maximum size of a set S such that no triangle is fully contained in S.
            # This is the independence number of the 3-uniform hypergraph of triangles.
            # There are algorithms for this, but n=18 is small.
            
            # Let's try to optimize the combination loop.
            # Only check sizes s where we haven't found a solution yet?
            # We start from n.
            # If we find one, we return.
            # Usually the max size is small (e.g., 11 or 12).
            # So we check s=18, 17, ..., 12.
            # For s=12, 18564 subsets.
            # For each subset, check against tris.
            # If tris is small, this is fast.
            
            for comb in itertools.combinations(range(n), s):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if mask contains any triangle
                is_tf = True
                for t in tris:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    return s
        return 0

    # Initialize
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    tris = get_triangles(adj)
    current_score = get_score(adj, tris)
    
    start_time = time.time()
    timeout = 110
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        u, v = random.sample(range(n), 2)
        edge_exists = (adj[u] >> v) & 1
        
        new_adj = adj[:]
        if edge_exists:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            # Check K4-free
            common = adj[u] & adj[v]
            valid = True
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                if (adj[k] & common) != 0:
                    valid = False
                    break
            if not valid:
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        new_tris = get_triangles(new_adj)
        new_score = get_score(new_adj, new_tris)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            adj = new_adj
            tris = new_tris
            current_score = new_score
            
        T *= cooling
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
