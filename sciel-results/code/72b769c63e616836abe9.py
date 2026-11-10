import itertools
import random
import time

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Returns a dictionary with 'n' and 'edges'.
    """
    def is_regular_subgraph(size, vertices, adj_mask):
        """
        Checks if the induced subgraph on 'vertices' is regular.
        'vertices' is a tuple of vertex indices.
        'adj_mask' is a list of integers where adj_mask[i] is the bitmask of neighbors of i.
        """
        if size < 6:
            return False
        
        # Count degrees within the subgraph
        degrees = [0] * size
        v_set = set(vertices)
        
        for i, u in enumerate(vertices):
            # For each vertex u in the subset, count neighbors in the subset
            deg = 0
            mask = adj_mask[u]
            for j, v in enumerate(vertices):
                if i != j and (mask >> v) & 1:
                    deg += 1
            degrees[i] = deg
        
        # Check if all degrees are equal
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def count_bad_subgraphs(n, adj_mask):
        """
        Counts the number of induced subgraphs of size >= 6 that are regular.
        """
        count = 0
        # Iterate over all subsets of size 6 to 22 is too slow for large n.
        # However, if we find a regular subgraph of size 6, any superset is NOT necessarily regular.
        # The condition is "No set of 6 or more vertices may induce a regular subgraph".
        # This means we must check ALL subsets of size k >= 6.
        
        # Optimization: If a subset of size 6 is regular, it contributes 1 to the count.
        # But we must also check size 7, 8, etc.
        
        # For n=20, total subsets is 2^20 ~ 1M. Checking all is feasible if the check is fast.
        # But we can prune: if a subset is regular, we don't need to check its supersets for regularity? 
        # No, a regular graph can have a non-regular supergraph, and vice versa.
        # Actually, if G[S] is regular, G[T] for T superset S is not necessarily regular.
        
        # Let's iterate all subsets of size >= 6.
        # For n=20, sum_{k=6}^{20} C(20,k) is very large (~1M).
        # Bitmask iteration is faster.
        
        # We will use a heuristic: just count size 6 and size 7 subsets primarily, 
        # as larger regular subgraphs are rarer in random-like graphs.
        # However, the definition requires checking ALL.
        
        # Let's optimize the check for a specific subset.
        
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                if is_regular_subgraph(size, subset, adj_mask):
                    count += 1
        return count

    def local_search(n, adj_mask, max_iters=5000):
        """
        Local search to minimize the number of bad subgraphs.
        """
        # Precompute list of vertices
        vertices = list(range(n))
        
        # Initial bad count
        current_bad = count_bad_subgraphs(n, adj_mask)
        
        if current_bad == 0:
            return adj_mask, 0
            
        for _ in range(max_iters):
            # Pick a random vertex to flip one of its edges
            # Better: Pick a random pair (u,v) and flip the edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u,v)
            old_u_mask = adj_mask[u]
            old_v_mask = adj_mask[v]
            
            if (old_u_mask >> v) & 1:
                # Remove edge
                adj_mask[u] = old_u_mask & ~(1 << v)
                adj_mask[v] = old_v_mask & ~(1 << u)
            else:
                # Add edge
                adj_mask[u] = old_u_mask | (1 << v)
                adj_mask[v] = old_v_mask | (1 << u)
                
            # Calculate delta in bad count? 
            # Recalculating from scratch is O(2^n * n^2) which is too slow for each step.
            # We need a faster delta calculation or a smaller n.
            
            # For n=20, 2^20 = 1,048,576. 
            # Counting all subsets of size >= 6 is ~1M combinations.
            # Python itertools.combinations is slow.
            
            # Let's use a different objective function that is cheaper to update,
            # or accept that we can only do a few restarts.
            
            # Alternative: Just check if the graph is valid at the end of a restart.
            # Use a faster validity check for n=20.
            
            # Let's try a simpler metric for the local search step: 
            # Number of regular triangles? No, we need size >= 6.
            
            # Given the time limit, a pure local search on the exact count is too slow.
            # Let's try a genetic algorithm or simulated annealing with a faster proxy?
            # Or just random restarts with a fast full check?
            
            # Let's implement a faster full check using bitmasks and precomputed subset properties?
            # Too complex for a scratch rewrite.
            
            # Strategy shift: 
            # 1. Start with n=20.
            # 2. Use a known good construction if possible? 
            # The prompt says "Graphs with 20 vertices are known".
            # Can I hardcode a known 20-vertex graph? 
            # I don't have one memorized. 
            # 
            # Let's try a heuristic: 
            # Random graphs G(n, p) with p=0.5.
            # Probability of a specific 6-set being regular is low.
            # Expected number of regular 6-sets: C(20,6) * P(regular).
            # P(regular 6-set) is roughly (number of regular graphs on 6 vertices) / 2^15.
            # There are many regular graphs. 
            # 
            # Let's just run random restarts and check validity.
            # Check validity for n=20:
            # Sum_{k=6}^{20} C(20,k) checks.
            # C(20,6) = 38760.
            # C(20,7) = 77520.
            # C(20,8) = 125970.
            # Total subsets >= 6 is approx 100,000 - 200,000.
            # For each subset, check regularity in O(k^2) or O(k).
            # With bitmasks, degree count is fast.
            # 200,000 * 36 ops ~ 7 million ops.
            # In Python, this might take 1-2 seconds per check.
            # We can do ~50-100 checks in 120s.
            # Random graphs might have very few or zero regular subgraphs.
            # If the expected number is < 1, there's a good chance a random graph works.
            
            # Let's estimate E[# regular 6-sets].
            # Number of regular graphs on 6 vertices:
            # 0-regular: 1
            # 1-regular: 3 (perfect matchings)
            # 2-regular: 6 (3+3 cycles, 4+2, 5+1? No, 2-reg on 6 is C6, C3+C3, C4+C2(impossible), so C6, C3+C3. Wait.
            # 2-regular graphs on 6 vertices:
            # - One cycle of length 6: (6-1)!/2 = 60? No. Number of cycles is (n-1)!/2. For n=6, 60.
            # - Two cycles of length 3: 1 way to partition into 2 sets of 3, each has 2 cycles? No, C3 has 2 orientations? No, undirected C3 is 1. 
            #   Number of ways to partition 6 into two 3-sets: C(6,3)/2 = 10. Each C3 is unique. So 10.
            # - Cycle of 4 and 2? 2-cycles don't exist in simple graphs.
            # - Cycle of 5 and 1? 1-vertex cycle doesn't exist.
            # So 2-reg: 60 + 10 = 70.
            # 3-regular:
            # - K_3,3: 1
            # - Triangular prism: 1
            # - Other?
            # Total regular graphs on 6 vertices is small, maybe < 100.
            # Total graphs on 6 vertices is 2^15 = 32768.
            # P(regular) ~ 100 / 32768 ~ 0.003.
            # E[#] = C(20,6) * 0.003 ~ 38760 * 0.003 ~ 116.
            # This is high. Random graphs will likely have ~100 regular 6-sets.
            # We need to break them.
            
            # Local search is necessary.
            # To make local search fast, we only update the count for subsets affected by the flip.
            # A flip of (u,v) affects all subsets containing both u and v, or subsets containing u and not v (degree of u changes), etc.
            # Actually, the regularity of a subset S depends on the degrees of vertices in S.
            # Flipping (u,v) changes degrees of u and v in every subset containing u or v.
            # This is complex to update incrementally.
            
            # Given the constraints, I will implement a simple random restart with a full check, 
            # but optimize the full check significantly.
            
            # Optimized Check:
            # Precompute for each vertex the bitmask of neighbors.
            # For a subset S (bitmask), compute degrees:
            # deg[v] for v in S is popcount(adj[v] & S).
            # We can iterate through subsets using Gosper's Hack or simple recursion?
            # For n=20, iterating 1M masks is okay.
            # For each mask, check if it has popcount >= 6.
            # If so, check if all popcount(adj[v] & mask) for v in mask are equal.
            
            # This is still O(2^n * n). 1M * 20 = 20M ops. ~1-2 seconds in Python.
            # So ~50-100 full checks per minute.
            # This is too slow for local search steps.
            
            # However, we can try a "greedy" construction or a specific known graph structure.
            # Let's try to construct a graph based on a known design, e.g., a Paley graph or similar.
            # Paley graph of order 19 is a known graph with no K4? No.
            # 
            # Let's try n=20.
            # I will implement a fast validity checker in C-speed (using numpy if possible? No, stdlib only).
            # 
            # Wait, I can use a heuristic: 
            # Minimize the number of regular 6-subsets.
            # If I can get 0 regular 6-subsets, I still need to check 7+.
            # But if there are no regular 6-subsets, it's less likely to have regular 7-subsets?
            # Not necessarily.
            
            # Let's assume that breaking all regular 6-subsets is the main hurdle.
            # I will perform a local search minimizing the count of regular 6-subsets.
            # Updating the count of regular 6-subsets is cheaper?
            # No, it's the same complexity.
            
            # Final Strategy:
            # 1. Try to find a graph with n=20.
            # 2. Use a time-bounded search.
            # 3. If found, return it.
            # 4. If not, return the best n found (maybe 19 or 18).
            
            # To make it faster, I will only check subsets of size 6 and 7.
            # If there are no regular 6 or 7 subsets, the probability of having a regular 8+ subset is low.
            # And I will verify the final candidate fully.
            
            pass

    # Implementation of the fast local search for n=20
    target_n = 20
    best_adj = None
    best_bad_count = float('inf')
    
    start_time = time.time()
    time_limit = 100  # Leave 20s for final check and return
    
    # We will try to minimize bad_6_7 count
    def count_bad_6_7(n, adj_mask):
        count = 0
        # Size 6
        for subset in itertools.combinations(range(n), 6):
            s_mask = 0
            for v in subset:
                s_mask |= (1 << v)
            # Check regularity
            degs = []
            for v in subset:
                degs.append((adj_mask[v] & s_mask).bit_count())
            if len(set(degs)) == 1:
                count += 1
        # Size 7
        for subset in itertools.combinations(range(n), 7):
            s_mask = 0
            for v in subset:
                s_mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj_mask[v] & s_mask).bit_count())
            if len(set(degs)) == 1:
                count += 1
        return count

    # Local search step
    def local_search_step(n, adj_mask):
        # Pick random u, v
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            return 0
        
        old_u = adj_mask[u]
        old_v = adj_mask[v]
        
        # Flip
        if (old_u >> v) & 1:
            adj_mask[u] = old_u & ~(1 << v)
            adj_mask[v] = old_v & ~(1 << u)
        else:
            adj_mask[u] = old_u | (1 << v)
            adj_mask[v] = old_v | (1 << u)
            
        return 1

    # Since full recalc is slow, we will do random restarts with random graphs and check.
    # But random graphs have ~100 bad 6-sets.
    # We need to fix them.
    
    # Let's try a "greedy" fix:
    # 1. Start with random graph.
    # 2. Find a bad 6-set.
    # 3. Flip a random edge inside or connected to it to break regularity.
    # 4. Repeat.
    
    found = False
    best_n = 5 # Fallback
    
    for trial in range(50): # 50 restarts
        if time.time() - start_time > time_limit:
            break
            
        n = 20
        adj_mask = [0] * n
        # Random graph p=0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        # Greedy Repair
        iterations = 0
        max_iterations = 500
        while iterations < max_iterations and time.time() - start_time < time_limit:
            # Find a bad 6-set
            bad_found = False
            # We need a fast way to find one. 
            # Randomly sample 6-sets until we find a bad one.
            for _ in range(100):
                subset = random.sample(range(n), 6)
                s_mask = 0
                for v in subset:
                    s_mask |= (1 << v)
                degs = [(adj_mask[v] & s_mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    # Found a bad 6-set
                    # Pick a random vertex in subset and flip a random edge
                    u = random.choice(subset)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    # Flip edge (u,v)
                    if (adj_mask[u] >> v) & 1:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                    else:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                    bad_found = True
                    break
            
            if not bad_found:
                # No bad 6-set found in 100 samples. Likely 0.
                # Verify fully?
                # Check size 6 and 7 fully to be sure.
                bad_count = count_bad_6_7(n, adj_mask)
                if bad_count == 0:
                    # Check larger sizes? 
                    # For now, assume 0 bad 6/7 implies valid. 
                    # We will do a full check on the final candidate.
                    found = True
                    best_adj = adj_mask[:]
                    best_n = n
                    break
                else:
                    # Continue searching for a specific bad set to fix
                    pass
            
            iterations += 1
            
        if found:
            break

    if found:
        # Full verification of the best candidate
        # Check all subsets >= 6
        valid = True
        n = best_n
        adj = best_adj
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                s_mask = 0
                for v in subset:
                    s_mask |= (1 << v)
                degs = [(adj[v] & s_mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # If invalid, try n=19?
        # For simplicity, if 20 fails, we return a smaller known valid graph or empty.
        # But 20 is known to exist. My search might have failed.
        # Let's fallback to n=10 with a random search which is easier.
        
    # Fallback: Find a valid graph with n=10 (easy)
    n = 10
    for _ in range(1000):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        valid = True
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                s_mask = 0
                for v in subset:
                    s_mask |= (1 << v)
                degs = [(adj_mask[v] & s_mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Last resort: Empty graph on 5 vertices
    return {'n': 5, 'edges': []}
