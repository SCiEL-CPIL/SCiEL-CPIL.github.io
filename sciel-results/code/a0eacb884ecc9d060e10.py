import numpy as np
import time
import itertools
import sys

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    # Generate pool of weight-4 vectors (norm^2 = 4)
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors.
    # Condition: min squared distance >= max squared norm.
    # If all have norm^2=4, we need dist^2 >= 4.
    # dist^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2(u.v) = 8 - 2(u.v).
    # 8 - 2(u.v) >= 4  =>  4 >= 2(u.v)  =>  u.v <= 2.
    # So we need a set of weight-4 vectors with pairwise dot product <= 2.
    
    def generate_pool():
        pool = []
        for idx in itertools.combinations(range(d), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool, dtype=np.int64)

    def solve_mis_bitset(n, adj, deadline):
        """
        Solve Maximum Independent Set on a graph with n vertices.
        adj[i] is a bitset (Python int) of neighbors of i.
        Returns the size of the MIS and the set of vertices.
        Uses a branch-and-bound algorithm with bitset operations.
        """
        # We want to find an independent set S such that |S| is maximized.
        # Conflict graph: edge if dot > 2.
        # Independent set in conflict graph = set with all dots <= 2.
        
        best_size = 0
        best_set = []
        
        # Precompute degrees for ordering? 
        # For Tomita's algorithm, we usually order vertices by degree.
        # Here we will use a simple recursive branch and bound with bitsets.
        
        # To make it faster, we can use a greedy coloring of the candidate set to bound the search.
        # But implementing greedy coloring with bitsets in Python is tricky.
        # Alternative: Use a simple recursive backtracking with strong pruning.
        
        # Let's use a standard MIS algorithm structure:
        # MIS(C, P) where C is candidate set, P is current independent set.
        # If C is empty, update best.
        # Else, pick a vertex v in C.
        # Branch 1: Include v. New C = C \ (N[v]). P = P U {v}.
        # Branch 2: Exclude v. New C = C \ {v}. P = P.
        
        # Optimization: Order vertices in C by degree in C? 
        # Or just iterate.
        
        # Since n=5280 is large, pure Python recursion might be too slow without strong bounds.
        # However, the graph is sparse. 
        # Let's try a simpler approach first: Greedy + Local Search with Bitsets.
        # The prompt suggests "branch-and-bound or exact ILP". 
        # Given the time limit (600s), an exact solver on 5280 nodes is risky.
        # But a very good heuristic using bitsets might outperform the current greedy.
        
        # Let's implement a fast bitset-based greedy with restarts and local search (1-swap, 2-swap).
        
        def greedy_init(seed):
            np.random.seed(seed)
            # Random permutation of vertices
            order = np.random.permutation(n)
            selected = []
            selected_set = set()
            # For fast check, we can maintain a bitset of compatible vertices?
            # Or just check against selected.
            # With bitsets, we can precompute "compatible" masks.
            # compat[i] = bitset of j such that dot(i,j) <= 2.
            
            # Wait, I should precompute the compatibility matrix as bitsets outside.
            pass

        # Let's restructure to precompute compatibility bitsets.
        # This is the key to speed.
        
        # Return placeholder if we decide not to use this specific function structure
        # We will implement the main logic in the outer scope for clarity.
        return 0, []

    # Precompute pool
    pool = generate_pool()
    n = len(pool)
    
    # Precompute dot products
    # M = pool @ pool.T
    # This is 5280x5280. 27 million floats. ~200MB. Acceptable.
    M = pool @ pool.T
    
    # Compatibility: dot <= 2
    # Note: M[i,i] = 4. We don't care about self.
    compat_mask = M <= 2
    
    # Convert to bitsets for fast operations?
    # Python integers can be used as bitsets.
    # compat_bits[i] is an int where bit j is 1 if i and j are compatible.
    # n=5280 bits. Python int handles this natively.
    
    compat_bits = [0] * n
    for i in range(n):
        bits = 0
        row = compat_mask[i]
        # Iterate over True indices? Or use np.packbits?
        # np.packbits is faster.
        packed = np.packbits(row, bitorder='little')
        # Convert bytes to int
        # Python int.from_bytes
        bits = int.from_bytes(packed, byteorder='little')
        compat_bits[i] = bits
        
    # Remove self from compat bits? 
    # For independent set, we don't want to select i if i is in the candidate set.
    # The standard algorithm removes N[v] (closed neighborhood) when adding v.
    # N[v] in conflict graph = {v} U {u | dot(v,u) > 2}.
    # So compatible(u,v) means NO edge in conflict graph.
    # If we add v, we remove all u such that NOT compatible(v,u).
    # i.e., we keep u such that compatible(v,u).
    # So the new candidate set C' = C & compat_bits[v].
    # And we must ensure v is not in C'? No, v is removed from C because it's selected.
    # Actually, standard MIS:
    # C is candidate set.
    # Pick v in C.
    # Branch 1: Add v. New C = (C \ {v}) & (Non-neighbors of v).
    # Since compat_bits[v] includes v (dot=4<=2 is false? Wait. 4<=2 is False. So self is NOT compatible).
    # Let's check: M[i,i] = 4. 4 <= 2 is False. So compat_bits[i] has 0 at bit i.
    # So C & compat_bits[v] automatically removes v.
    # Perfect.
    
    def greedy_mis_bitset(seed):
        np.random.seed(seed)
        C = (1 << n) - 1  # All vertices
        selected = []
        
        # Greedy: pick vertex with max degree in C? 
        # Or random?
        # Random is faster to implement. Let's try random first.
        # To pick random from C:
        # Generate random index, check if in C. 
        # Better: extract bits, pick random one.
        
        while C:
            if time.time() > deadline:
                break
            
            # Pick a vertex v from C
            # Get list of set bits? Too slow if C is large.
            # Use a random mask?
            # Simple approach: iterate through bits? 
            # Let's just pick a random integer r in [0, n-1] and check if C has bit r.
            # If n=5280, and C is sparse, this is slow.
            # If C is dense, it's fast.
            
            # Alternative: Maintain a list of available vertices? 
            # Bitset is good for intersection, bad for random selection.
            
            # Let's use a simple heuristic: Pick the vertex in C with the smallest index?
            # That's deterministic.
            # Let's try: Pick v = (C & -C).bit_length() - 1 (lowest set bit)
            # This is fast.
            v = (C & -C).bit_length() - 1
            
            # Add v to selected
            selected.append(v)
            
            # Update C: C = C & compat_bits[v]
            C &= compat_bits[v]
            
        return selected

    def local_search_bitset(selected, deadline):
        """
        1-swap local search.
        Try to replace a selected vertex with an unselected one to increase size?
        Actually, if we are maximizing size, we can't just swap 1 for 1 and increase size.
        We need to find an augmenting path or just try to add more vertices?
        Greedy MIS doesn't necessarily find the max.
        
        Better: Try to find a vertex NOT in selected that is compatible with ALL selected.
        If found, add it.
        This is just "can we add more?".
        
        The greedy algorithm stops when C is empty.
        C is the set of vertices compatible with ALL selected.
        So if C is empty, we can't add any more.
        
        To improve, we need to swap.
        Remove one selected vertex v, see if we can add multiple new ones?
        Or swap v with u if u allows more future additions?
        
        Given time, let's stick to multiple restarts of the greedy bitset algorithm.
        It's very fast.
        """
        pass

    best_size = 0
    best_set = []
    
    # Run many restarts of greedy
    # Each restart is O(n^2 / word_size) roughly? 
    # n=5280. n^2 = 27M. /64 = 400k ops. Very fast.
    # We can run thousands of restarts.
    
    num_restarts = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Vary the selection order
        # We can randomize the "lowest set bit" choice by permuting the vertex labels?
        # Or just use different seeds for a random selection strategy.
        
        # Let's implement a random selection strategy for the greedy step.
        # To pick a random vertex from C:
        # We can use a random number to pick a position in the list of set bits.
        # But getting the list of set bits is O(popcount(C)).
        
        # Alternative: Randomize the vertex order once, then greedy picks the first available in that order.
        # This is equivalent to a specific greedy order.
        
        # Let's do this:
        # 1. Generate a random permutation of vertices.
        # 2. Iterate through permutation.
        # 3. If vertex is in C, select it, update C.
        
        perm = np.random.permutation(n)
        C = (1 << n) - 1
        selected = []
        
        for v in perm:
            if C & (1 << v):
                selected.append(v)
                C &= compat_bits[v]
                if not C:
                    break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    # Local search improvement on best_set
    # Try to swap vertices to increase size?
    # Actually, if we found a set of size K, can we find K+1?
    # We can try to "relax" the set.
    
    # For now, return the best greedy result.
    # 5280 nodes, greedy is very fast. 1000 restarts should take < 1 second.
    # We have 580 seconds!
    # We can do much more sophisticated search.
    
    # Let's implement a simple 1-swap local search.
    # Current set S.
    # For each v in S, remove it.
    # Try to add any u not in S that is compatible with S\{v}.
    # If we can add u, and potentially more, we might improve.
    # But simple 1-for-1 swap doesn't increase size.
    
    # To increase size, we need to find a vertex u not in S compatible with S.
    # The greedy algorithm ensures that at the end, no such u exists.
    # So we must swap.
    # If we remove v, maybe we can add u1, u2, u3?
    
    # Let's try a limited depth search for improvement.
    
    def try_improve(S, deadline):
        S_set = set(S)
        S_list = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, v in enumerate(S_list):
                if time.time() > deadline:
                    break
                # Remove v
                S_temp = S_list[:i] + S_list[i+1:]
                S_temp_set = set(S_temp)
                
                # Find candidates compatible with S_temp
                # C = intersection of compat_bits[u] for u in S_temp
                C = (1 << n) - 1
                for u in S_temp:
                    C &= compat_bits[u]
                
                # Remove already selected
                for u in S_temp_set:
                    C &= ~(1 << u)
                
                # If C is not empty, we can add a vertex.
                if C:
                    # Pick one? Or try to maximize?
                    # Let's just pick one random one from C to swap with v.
                    # This keeps size same.
                    # To increase size, we need to find if we can add MORE than 1.
                    # This is getting complex.
                    
                    # Let's just try to find if there is ANY vertex in C.
                    # If yes, we can swap v with u.
                    # This doesn't increase size, but might lead to a better position for future steps?
                    # Actually, if we just want max size, and greedy failed, 
                    # it means the "independent set" is maximal but not maximum.
                    
                    # Let's try a different approach:
                    # Use the bitset C to see if we can add a vertex.
                    # If we remove v, and C has > 1 bits, maybe we can add 2?
                    
                    # For now, let's just accept that greedy is good enough with many restarts.
                    # The graph is vertex-transitive and regular?
                    # Weight 4 vectors.
                    # Degree of conflict graph:
                    # How many weight 4 vectors have dot > 2 with a fixed weight 4 vector?
                    # Fixed v has support S (size 4).
                    # u has support T (size 4).
                    # dot = sum_{i in S∩T} v_i u_i.
                    # Max dot is 4 (if S=T, signs same).
                    # Dot > 2 means dot is 3 or 4.
                    # If S=T:
                    #   signs same: dot=4. (1 vector)
                    #   1 sign diff: dot=2. (4 vectors) -> Compatible
                    #   2 sign diff: dot=0. (6 vectors) -> Compatible
                    #   3 sign diff: dot=-2. (4 vectors) -> Compatible
                    #   4 sign diff: dot=-4. (1 vector) -> Compatible
                    # So for S=T, only 1 conflict.
                    
                    # If |S∩T|=3:
                    #   3 common indices. 1 unique in S, 1 unique in T.
                    #   dot = sum of 3 products.
                    #   Max sum 3 (all +1). Min -3.
                    #   Values: 3, 1, -1, -3.
                    #   Dot > 2 means Dot=3.
                    #   This happens if the 3 common signs are all same.
                    #   Number of such u:
                    #   Choose 3 indices from 4 in S: 4 ways.
                    #   Choose 1 index from 7 not in S: 7 ways.
                    #   Signs of common 3 must match v.
                    #   Sign of new index: 2 ways.
                    #   Total: 4 * 7 * 2 = 56 conflicts.
                    
                    # If |S∩T|=2:
                    #   Max dot 2. Not > 2.
                    
                    # So total conflicts per vertex:
                    # 1 (same support, same sign) + 56 (3 overlap, 3 match) = 57.
                    
                    # Graph is 57-regular on 5280 vertices.
                    # Very sparse!
                    # Greedy should work very well.
                    
                    pass
        
        return S_list

    # Apply local search to best_set
    # Actually, let's just run more restarts. 1000 is small.
    # Let's run 10000 restarts.
    
    for r in range(1000, 10000):
        if time.time() > deadline - 5:
            break
        
        perm = np.random.permutation(n)
        C = (1 << n) - 1
        selected = []
        
        for v in perm:
            if C & (1 << v):
                selected.append(v)
                C &= compat_bits[v]
                if not C:
                    break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

    if best_size == 0:
        # Fallback
        return np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)], dtype=np.int64)

    result = pool[best_set]
    return np.around(result).astype(np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Found {len(res)} points")
