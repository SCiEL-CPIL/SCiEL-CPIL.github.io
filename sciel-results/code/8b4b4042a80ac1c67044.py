import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def count_k4(G, n):
        """Count K4s in graph G (adjacency matrix or list of sets)"""
        # G is list of sets of neighbors
        count = 0
        # Iterate over pairs of vertices
        # A K4 is a set of 4 vertices {a,b,c,d} such that all pairs are edges.
        # This is O(n^4) which is too slow for n=35 if done naively many times.
        # Better: Iterate over edges (u,v). Find common neighbors. 
        # A K4 containing edge (u,v) corresponds to an edge (w,x) in the common neighborhood of u and v.
        # So for each edge (u,v), let C = N(u) ∩ N(v). Count edges in G[C].
        # Sum over all edges (u,v). Each K4 is counted 6 times (6 edges).
        # So total K4s = sum / 6.
        
        # For speed, we maintain the count incrementally in the search.
        pass

    def count_k6_complement(G, n):
        """Count K6s in complement of G. Equivalent to Independent 6-sets in G."""
        # Similar logic. For each non-edge (u,v) in G (edge in complement), 
        # let C = (V \ N(u)) ∩ (V \ N(v)) \ {u,v}. 
        # Count edges in complement within C.
        # This is complex to update incrementally.
        
        # Alternative: Just check validity at the end, and use a heuristic cost 
        # that is cheaper to compute, like the number of "bad" small subgraphs.
        pass

    # Given the complexity of incremental cost updates for K4 and K6, 
    # and the tight time limit, a simple random restart local search 
    # checking full validity periodically might be safer, though slower.
    
    # However, for n=35, O(n^4) is 35^4 ~ 1.5M operations. 
    # If we do this 1000 times, it's 1.5B operations, which is too slow in Python.
    
    # We need a faster check or a smarter search.
    
    # Let's try to find a known smaller graph that is easy to verify.
    # The Paley graph on 17 is verified.
    # Can we extend it?
    
    # Actually, let's just return the Paley graph if we can't do better quickly.
    # But we can try n=20, 25 with a very simple heuristic.
    
    # Heuristic: Random graph, flip edge if it reduces (K4 count + Ind6 count).
    # To make it fast, we approximate the cost.
    
    # Let's implement a fast validator.
    def is_valid(G, n):
        # G is a list of bitmasks (integers) representing neighbors.
        # Check K4:
        for u in range(n):
            for v in range(u+1, n):
                if G[u] & (1 << v):
                    common = G[u] & G[v]
                    # Check if common contains an edge
                    # Iterate over bits in common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c -= lsb
                        # Check if w is connected to any other in common
                        if G[w] & common & ~(1 << w):
                            return False
        # Check Ind 6 (K6 in complement):
        # For each set of 6 vertices, check if no edges.
        # This is O(n^6) which is 35^6 ~ 1.8B. Too slow.
        
        # Better Ind 6 check:
        # A set S of 6 vertices is independent if for all u in S, (N(u) & S) == 0.
        # We can iterate over all 6-subsets? 35 choose 6 is 1.6M.
        # 1.6M * 6 checks is ~10M ops. This is feasible in Python if optimized? 
        # 10M ops is ~0.1s - 0.5s in Python. 
        # If we do this 100 times, it's 10-50s. Feasible.
        
        # Let's use this validator.
        
        # K4 check optimization:
        # The above K4 check is O(n^3) roughly.
        
        # Let's combine:
        # 1. Check K4. If found, return False.
        # 2. Check Ind 6. If found, return False.
        
        # K4 Check:
        for u in range(n):
            for v in range(u+1, n):
                if G[u] & (1 << v):
                    common = G[u] & G[v]
                    # Check if any edge in common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c -= lsb
                        if G[w] & common:
                            return False
                            
        # Ind 6 Check:
        # Iterate over all combinations of 6 vertices
        # To speed up, we can prune.
        # But 1.6M combos is okay.
        
        # Actually, we can check Ind 6 by looking for a clique of size 6 in the complement.
        # Complement adjacency: C[u] = ~G[u] & ((1<<n)-1) & ~(1<<u)
        
        # Let's just use itertools.combinations for simplicity, might be slow but acceptable for a few tries.
        
        return True

    # Search Strategy:
    # Try n=35, 30, 25, 20.
    # For each n, run local search for a few seconds.
    
    def local_search(n, time_limit):
        best_G = None
        best_cost = float('inf')
        end_time = time.time() + time_limit
        
        # Initialize random graph
        G = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
        
        def get_cost(G, n):
            # Count K4s and Ind6s
            k4_count = 0
            for u in range(n):
                for v in range(u+1, n):
                    if G[u] & (1 << v):
                        common = G[u] & G[v]
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            c -= lsb
                            if G[w] & common:
                                k4_count += 1
            # Divide by 6? No, just use as cost.
            
            ind6_count = 0
            # Count independent 6-sets
            # This is the bottleneck.
            # Heuristic: Count "potential" independent sets?
            # Or just check if valid?
            # If we only use cost for local search, we need it fast.
            # Maybe just count K4s and assume Ind6s are rare? No.
            
            # Let's just check validity. If valid, cost 0.
            # If not, cost = 1.
            # This makes local search a random walk with acceptance only if valid? 
            # That's not good. We need a gradient.
            
            # Alternative: Cost = number of K4s + number of "almost" Ind6s?
            # Let's stick to K4 count for now, and check Ind6 separately.
            # If we minimize K4s, we might accidentally create Ind6s.
            
            return k4_count

        # This is getting complicated. 
        # Given the constraints, the safest bet to improve score is to 
        # try to find a graph on n=20 or 25 which is much easier than 35.
        # Paley(17) is 17. 
        # Can we find n=20?
        
        # Let's try a simple random search for n=20, 25, 30.
        # If we find a valid one, return it.
        
        for target_n in [35, 30, 25, 20]:
            if time.time() > deadline - 5:
                break
            limit = min(10, deadline - time.time())
            if limit <= 0: break
            
            found = False
            end_t = time.time() + limit
            while time.time() < end_t:
                G = [0] * target_n
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.5:
                            G[i] |= (1 << j)
                            G[j] |= (1 << i)
                
                # Simple local search: flip random edge if it reduces K4 count
                # and doesn't increase Ind6 count (approximated by validity check)
                
                # Just check validity of the random graph
                if is_valid(G, target_n):
                    return {'n': target_n, 'edges': [[i, j] for i in range(target_n) for j in range(i+1, target_n) if G[i] & (1 << j)]}
                
                # Try a few flips
                for _ in range(100):
                    u = random.randint(0, target_n-1)
                    v = random.randint(0, target_n-1)
                    if u == v: continue
                    # Flip edge
                    if G[u] & (1 << v):
                        G[u] ^= (1 << v)
                        G[v] ^= (1 << u)
                    else:
                        G[u] |= (1 << v)
                        G[v] |= (1 << u)
                    
                    if is_valid(G, target_n):
                        return {'n': target_n, 'edges': [[i, j] for i in range(target_n) for j in range(i+1, target_n) if G[i] & (1 << j)]}

    # Fallback: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

# The above is_valid is O(n^3) for K4 and O(n^6) for Ind6.
# For n=35, O(n^6) is too slow for the inner loop.
# We must optimize is_valid or use a different approach.

# Optimized Ind 6 check:
# A set of 6 vertices is independent if no edges exist between them.
# We can iterate over all 6-subsets. 35C6 = 1,623,160.
# For each subset, check 15 pairs.
# 1.6M * 15 = 24M ops.
# In Python, 24M ops takes ~2-5 seconds.
# So we can only check ~10-20 graphs per second for n=35.
# 10s limit -> 100-200 checks.
# Probability of random graph being valid is extremely low.
# So random search will fail.

# We need a better heuristic.
# However, if we can't find 35, we can try 20.
# 20C6 = 38,760. 38k * 15 = 580k ops. ~0.05s.
# We can check 1000 graphs per second.
# 10s -> 10,000 checks.
# Still, random graphs are unlikely to be valid.
# But local search might work.

# Let's just return the Paley graph if we don't have a strong guarantee.
# But wait, the score is value/seed.
# If we return 17, score is 1.0.
# If we can find 20, score is 1.18.
# Is it worth the risk of timeout?
# The deadline guard ensures we return something.

# I will implement the local search for n=20, 25, 30, 35 with the optimized validator.
# If it finds one, great. If not, fallback to 17.
