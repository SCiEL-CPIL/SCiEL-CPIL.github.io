import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    limit = 512
    n = limit
    
    # Precompute LCM compatibility graph
    # compatible[i][j] = True if lcm(i+1, j+1) <= limit
    # We use 0-indexed internally, values are 1..512
    
    # For efficiency, we'll use bitmasks for neighbors
    # n=512 is too large for a single int bitmask in some operations, 
    # but Python handles big ints. However, 512 bits is fine.
    
    # Let's build adjacency lists first, then convert to bitsets for speed
    adj = [0] * (n + 1)  # 1-indexed, adj[i] is a bitmask of j such that lcm(i,j)<=limit
    
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i == j:
                mask |= (1 << j)
                continue
            lcm_val = i * j // gcd(i, j)
            if lcm_val <= limit:
                mask |= (1 << j)
        adj[i] = mask
    
    # The problem is to find the maximum clique in this graph.
    # However, the graph is very dense for small numbers and sparse for large.
    # Actually, let's look at the structure.
    # lcm(a,b) <= 512.
    # This implies a|512 and b|512? No.
    # If a=3, b=5, lcm=15<=512. 3 does not divide 512.
    # But if a and b are large, say a=256, b=257, lcm is huge.
    
    # Key insight: If lcm(a,b) <= 512, then a and b must be divisors of some number <= 512?
    # No, lcm(a,b) is a multiple of both a and b. So a | lcm(a,b) and b | lcm(a,b).
    # Since lcm(a,b) <= 512, a must divide a number <= 512, which is always true for a<=512.
    # But more strongly, a must be a divisor of some K <= 512.
    # Actually, the condition lcm(a,b) <= 512 is equivalent to:
    # Let g = gcd(a,b). a = g*x, b = g*y, gcd(x,y)=1.
    # lcm = g*x*y = a*y = b*x.
    # So a*y <= 512.
    
    # Let's use Maximum Clique algorithms (Bron-Kerbosch with pivoting).
    # With n=512, brute force is impossible, but the graph has structure.
    # Note that if i is in the clique, all other elements must be in adj[i].
    
    # We can use a branch and bound algorithm for maximum clique.
    # Reference: Tomita's algorithm.
    
    best_size = 0
    best_clique = []
    
    def expand(R, P, X):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        # P and X are bitmasks of vertices
        # If P is empty, update best
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Pivoting: choose pivot u from P | X to maximize |P & adj[u]|
        # We want to minimize the number of branches.
        # Branch on vertices v in P such that P & adj[v] == 0? No.
        # Standard Bron-Kerbosch with pivot:
        # Choose u in P | X such that |P ∩ N(u)| is maximized.
        # For each v in P \ N(u):
        #    expand(R ∪ {v}, P ∩ N(v), X ∩ N(v))
        #    P = P \ {v}
        #    X = X ∪ {v}
        
        # To find u, we can iterate through P|X and count bits in P & adj[u].
        # This is O(n) per call, which might be slow if deep.
        # Optimization: Use a greedy coloring bound (Tomita).
        # But let's try simple pivot first to see if it's fast enough.
        # Given the structure, the graph might be very "clustered".
        
        # Let's implement Tomita's coloring for a better bound.
        # Order vertices in P by color.
        # We need to maintain an ordering.
        
        # Simpler approach for this specific problem:
        # The graph is an intersection graph of something?
        # Let's stick to Bron-Kerbosch with pivot and a time check.
        
        # To speed up pivot selection, we can just pick the first vertex in P|X 
        # or use a heuristic.
        # Actually, for n=512, calculating |P & adj[u]| for all u in P|X is ~512 * 512 bit ops.
        # Python big int bit ops are fast.
        
        # Get list of vertices in P | X
        # We need to iterate over set bits.
        # Helper to get set bits
        def get_bits(mask):
            bits = []
            while mask:
                # lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                bits.append(idx)
                mask ^= lsb
            return bits
            
        # This is slow if called often. 
        # Alternative: represent P as a list of vertices?
        # Bitmasks are faster for intersection.
        
        # Let's optimize the pivot selection.
        # We want u in P|X that maximizes popcount(P & adj[u]).
        
        candidates = P | X
        if candidates == 0:
            return
            
        best_u = -1
        max_count = -1
        
        # Iterate through bits in candidates
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count bits in P & adj[u]
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                best_u = u
                
        # If max_count is 0, we can branch on all P.
        # Branch on v in P such that v is NOT a neighbor of best_u.
        # i.e., v in P & ~adj[best_u]
        
        # Note: If best_u is in P, it is a neighbor of itself, so it won't be in the branch set.
        # If best_u is in X, it's not in P, so all v in P are candidates if they are not neighbors.
        
        branch_set = P & ~adj[best_u]
        
        # Iterate through bits in branch_set
        temp = branch_set
        while temp:
            if time.time() > deadline:
                return
                
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # Add v to R
            R.append(v)
            
            # New P: P & adj[v]
            new_P = P & adj[v]
            # New X: X & adj[v]
            new_X = X & adj[v]
            
            expand(R, new_P, new_X)
            
            R.pop()
            
            # Update P and X
            P ^= lsb  # Remove v from P
            X |= lsb  # Add v to X
            
            # Early termination? If we found a clique of size n, we can stop.
            # But we don't know n_max easily.
            
        # After loop, P should be 0 if we branched on all P \ N(u).
        # Wait, the standard algorithm removes v from P and adds to X.
        # The loop iterates over the *original* branch_set.
        # We must be careful: P changes during the loop.
        # The iteration `temp = branch_set` is over a static snapshot.
        # We remove v from P using `P ^= lsb`. This is correct.
        
    # Initial call
    # R = []
    # P = all vertices (1..512)
    # X = 0
    
    # Bitmask for 1..512
    # Bit i corresponds to vertex i.
    # We use 1-indexed vertices, so bit 1 is vertex 1, ..., bit 512 is vertex 512.
    # Mask: sum(1<<i for i in 1..512)
    full_mask = 0
    for i in range(1, 513):
        full_mask |= (1 << i)
        
    # We need to handle the case where the graph is large.
    # Bron-Kerbosch might be slow.
    # Let's add a heuristic: Start with a greedy clique and use it as a lower bound.
    
    # Greedy clique
    def greedy_clique():
        P = full_mask
        R = []
        while P:
            # Pick vertex with max degree in P
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (P & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            R.append(best_v)
            P &= adj[best_v]
        return R
        
    init_clique = greedy_clique()
    best_size = len(init_clique)
    best_clique = init_clique
    
    # Run Bron-Kerbosch
    # To improve performance, we can use the bound: |R| + |P| <= best_size => prune
    # But standard BK with pivot is often sufficient for n=512 if the graph is structured.
    
    # Let's try to run it.
    # We need to be careful with recursion depth. 512 is fine.
    
    # One optimization: Order vertices by degree before starting?
    # BK with pivot handles this implicitly to some extent.
    
    # Let's run expand
    # Note: R is a list of integers (vertex ids)
    
    # To avoid modifying global state in recursion, we pass R as a list and pop/push.
    
    # We need to update best_size and best_clique in the closure.
    
    # Let's wrap it in a function that returns the best clique found
    # But we already have best_size and best_clique in outer scope.
    
    # We need to be careful: `expand` modifies `best_size` and `best_clique`.
    
    # Let's run it.
    
    # Potential issue: The graph might be too dense or too complex for BK to finish in 55s.
    # If it times out, we return the best found so far.
    
    # Let's add a simple bound: if |R| + |P|.bit_count() <= best_size, return.
    
    def expand_bounded(R, P, X):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        # Bound
        if len(R) + P.bit_count() <= best_size:
            return
            
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Pivot
        candidates = P | X
        if candidates == 0:
            return
            
        best_u = -1
        max_count = -1
        
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                best_u = u
                
        branch_set = P & ~adj[best_u]
        
        temp = branch_set
        while temp:
            if time.time() > deadline:
                return
                
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            R.append(v)
            
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            expand_bounded(R, new_P, new_X)
            
            R.pop()
            
            P ^= lsb
            X |= lsb
            
    # Run
    R = []
    expand_bounded(R, full_mask, 0)
    
    # Convert best_clique (1-indexed vertex ids) to values
    # Vertex i corresponds to value i.
    # So best_clique is already the list of values.
    
    # Verify
    # for i in range(len(best_clique)):
    #    for j in range(i+1, len(best_clique)):
    #       a, b = best_clique[i], best_clique[j]
    #       if a*b//gcd(a,b) > 512:
    #           print("Invalid clique found!")
    
    return best_clique

