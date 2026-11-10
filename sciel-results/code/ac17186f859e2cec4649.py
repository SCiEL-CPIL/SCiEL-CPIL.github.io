import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Greedy construction with repair (stochastic local search).
    We build the graph vertex by vertex. For each new vertex, we choose neighbors
    to avoid creating any forbidden regular induced subgraph that includes the new vertex.
    If stuck, we use a repair phase to fix conflicts.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n=20. Known to exist.
    # We will use bitsets for adjacency to allow fast subset degree checks.
    
    # Precompute popcount for 16-bit chunks? Python int.bit_count() is fast in C.
    
    def find_conflict(n, adj):
        """
        Finds a regular induced subgraph of size >= 6.
        Returns (mask, deg) if found, else (None, None).
        adj is a list of int bitmasks.
        """
        # We only need to check subsets of size 6? 
        # If a subset of size k > 6 is regular, any subset of size 6 within it is not necessarily regular.
        # Wait. If S is regular of size k, all degrees are k-1 or 0? No, regular means all degrees in S are equal to d.
        # If we take a subset T of S of size 6, the degrees in T are degrees in S restricted to T.
        # They are NOT necessarily equal.
        # Example: C5 is regular of size 5 (deg 2). Any subset of size 4 is a path or cycle? 
        # Actually, the condition is "No set of 6 or more vertices may induce a regular subgraph".
        # This implies we must check ALL subsets of size >= 6.
        # However, if a subset S of size 7 is regular, it's a violation.
        # Is it sufficient to check only size 6?
        # No. A 7-vertex regular graph (e.g. 3-regular) might not contain a 6-vertex regular subgraph.
        # Example: K_{3,3} minus a perfect matching? No, that's 6 vertices.
        # Consider a 7-vertex 3-regular graph. Does it always contain a 6-vertex regular subgraph?
        # Not necessarily.
        # So we must check all k >= 6.
        
        # Optimization: 
        # If we check all subsets of size 6 and find none, do we still need to check larger ones?
        # Yes.
        
        # Strategy: 
        # 1. Check all subsets of size 6.
        # 2. If none found, check size 7, etc.
        
        for k in range(6, n + 1):
            for S in combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                # Check if induced subgraph on mask is regular
                # Degrees in S:
                # deg(v) = popcount(adj[v] & mask)
                
                # Optimization: Early break if degrees differ.
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return mask, first_deg
        return None, None

    def is_valid(n, adj):
        mask, deg = find_conflict(n, adj)
        return mask is None

    # Greedy Construction
    def greedy_construct(n):
        adj = [0] * n
        
        for v in range(n):
            if v == 0:
                continue
            
            # We want to choose a neighbor set N for v.
            # N is a subset of {0, ..., v-1}.
            # Constraint: No subset S containing v with |S| >= 6 is regular.
            
            # This is a hard constraint satisfaction problem.
            # Heuristic: Random sampling of N, then verification.
            
            # To speed up, we can check "critical" subsets.
            # But for simplicity and correctness, let's try random Ns and check full validity 
            # or incremental validity.
            
            # Incremental check:
            # Adding v with neighbor set N.
            # New conflicts must involve v.
            # So we check all subsets S such that v in S, |S| >= 6.
            # Let T = S \ {v}. T is subset of {0..v-1}. |T| >= 5.
            
            # We try up to 50 random Ns.
            found_valid_N = False
            for _ in range(50):
                if time.time() > deadline:
                    return None
                
                # Generate random N
                # Bias: maybe uniform random is fine.
                N = 0
                for u in range(v):
                    if random.random() < 0.5:
                        N |= (1 << u)
                
                # Check if adding v with N is valid w.r.t existing vertices
                # We need to check all T subset of {0..v-1} with |T| >= 5.
                # For each T, S = T U {v}.
                # deg_S(v) = popcount(N & T)
                # deg_S(u) for u in T = popcount(adj[u] & T) + (1 if u in N else 0)
                
                # This check is expensive if done for ALL T.
                # Number of T of size >= 5 in {0..v-1}.
                # If v=10, size of set is 10. C(10,5)+C(10,6)+... = 252+210+120+45+10+1 = 638.
                # For each T, check regularity.
                # This is manageable for small v.
                # For v=19, size 19. Sum_{k=5}^{19} C(19,k) is huge (~500k).
                # Checking 500k subsets * 50 tries = 25M checks. Too slow.
                
                # Alternative:
                # Just do the full check on the current graph (including v) if we add v.
                # But that's O(2^v).
                
                # Better Alternative:
                # Use a "conflict-based" rejection.
                # Only check subsets that are LIKELY to be regular?
                # Or, just accept a small risk and verify at the end?
                # If we verify at the end and it fails, we backtrack?
                
                # Let's try a simpler approach:
                # Generate the whole graph randomly.
                # Verify.
                # If invalid, repair.
                
                # Repair:
                # Find a conflict (regular subgraph S).
                # Modify the graph to break regularity in S.
                # How? Flip an edge within S or incident to S?
                # If we flip an edge (u,v) in S, degrees of u and v change by +-1.
                # This might break the regularity.
                
                # This is a standard local search.
                # 1. Start with random graph.
                # 2. While invalid:
                #    a. Find a minimal conflict S (size 6 preferred for speed).
                #    b. Pick an edge (u,v) in S or incident to S? 
                #       Actually, changing an edge inside S is most direct.
                #       Let's pick a random edge inside S and flip it.
                #    c. Update graph.
                
                # This might work!
                
                found_valid_N = True # We will break out and use the repair strategy
                break
            
            # Since we decided to use repair strategy, let's implement that.
            # The greedy construction above is a dead end for now.
            # Let's switch to:
            # 1. Generate random graph on n vertices.
            # 2. Loop:
            #    a. Find conflict.
            #    b. If no conflict, return.
            #    c. If timeout, break.
            #    d. Pick a conflict S.
            #    e. Flip a random edge in S (or incident to S if S is too rigid).
            #       Flipping edge inside S is best.
            #    f. Update adj.
            
            # Let's implement this repair strategy directly in the main flow.
            # We don't need the greedy construction loop if we do repair.
            # So I will abort the greedy construction and use repair.
            
            return None # Signal to use repair strategy

    # Repair Strategy
    def repair_search(n, max_iters=10000):
        # Generate initial random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Find conflict
            conflict_mask, deg = find_conflict(n, adj)
            
            if conflict_mask is None:
                # Valid!
                return adj
            
            # Flip a random edge within the conflict
            # Get vertices in conflict
            verts = []
            m = conflict_mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # Pick two distinct vertices from verts
            if len(verts) < 2:
                # Should not happen for size >= 6
                continue
                
            u, v = random.sample(verts, 2)
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return None

    # Try to find a valid graph on n=20
    # Use repair search
    
    # To improve success rate, we can try different initial densities?
    # Or just run repair for a while.
    
    # Let's run repair for n=20
    for attempt in range(5): # 5 restarts with different random seeds
        random.seed(attempt * 12345)
        adj = repair_search(20, max_iters=5000)
        if adj is not None:
            # Convert to edge list
            edges = []
            for i in range(20):
                for j in range(i+1, 20):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            
            # Final verification
            if is_valid(20, adj):
                return {'n': 20, 'edges': edges}
    
    # If n=20 failed, try n=19
    for attempt in range(5):
        random.seed(1000 + attempt * 12345)
        adj = repair_search(19, max_iters=5000)
        if adj is not None:
            edges = []
            for i in range(19):
                for j in range(i+1, 19):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            if is_valid(19, adj):
                return {'n': 19, 'edges': edges}
    
    # Fallback to smaller n
    for n in range(18, 5, -1):
        random.seed(2000 + n * 12345)
        adj = repair_search(n, max_iters=1000)
        if adj is not None:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            if is_valid(n, adj):
                return {'n': n, 'edges': edges}
    
    # Last resort: trivial graph
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
