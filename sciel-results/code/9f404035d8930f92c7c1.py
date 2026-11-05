import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    # Branch and Bound for Maximum Independent Set
    # Uses greedy coloring bound (Tomita style) which is very effective
    def branch_and_bound(initial_mask, max_time=5.0):
        # We are solving for MIS on the subgraph induced by initial_mask
        # But we want to find a set that is an independent set in the full graph.
        # So we work on the graph restricted to vertices in initial_mask.
        
        # Precompute adjacency restricted to initial_mask?
        # Actually, adj[v] already contains all neighbors.
        # We just need to ensure we only pick from initial_mask.
        
        best_size = 0
        best_set = 0
        
        # Tomita's algorithm usually works on a list of vertices.
        # We can use bitmasks for efficiency.
        
        # We need a coloring function for the subgraph induced by a set of vertices.
        # Greedy coloring:
        # 1. Order vertices (e.g., by degree in the subgraph)
        # 2. Assign colors greedily.
        # 3. The number of colors is an upper bound on the size of the independent set?
        # No, the number of colors is an upper bound on the size of the CLIQUE.
        # Wait.
        # In MIS, we want a set of non-adjacent vertices.
        # In the complement graph, this is a Clique.
        # Tomita's algorithm solves Maximum Clique.
        # Max Independent Set in G is Max Clique in G_complement.
        
        # Let's switch perspective:
        # Construct G_complement on the candidate set?
        # Or just implement MIS B&B directly.
        
        # Standard MIS B&B:
        # Bound: The size of the maximum independent set in a graph is at most
        # the number of colors in a greedy coloring of the graph?
        # No. The chromatic number is a lower bound on the clique number of the complement.
        # The clique number of the complement is the independent set number of the original.
        # So, if we color the ORIGINAL graph, the number of colors is NOT an upper bound on the MIS.
        # If we color the COMPLEMENT graph, the number of colors is an upper bound on the Clique (which is MIS in original).
        
        # So we need to compute a greedy coloring of the COMPLEMENT graph induced by the candidate set.
        # Complement edge (u,v) exists if u and v are NOT adjacent in G.
        # i.e., (adj[u] & (1<<v)) == 0.
        
        # Let's implement a simple greedy coloring for the complement graph.
        # Given a set of vertices P (bitmask), find a coloring.
        # To make the bound tight, we order vertices by degree in the complement (which is N-1 - degree_in_G).
        # Low degree in G -> High degree in Complement -> Harder to color with few colors?
        # Actually, for Max Clique (MIS), we want a tight upper bound.
        # Tomita uses "color sort" where vertices are colored greedily, and the color number is the bound.
        # The order matters. Usually, vertices are ordered by descending degree in the graph being solved (Clique graph = Complement).
        
        # Let's implement:
        # 1. Extract list of vertices from mask.
        # 2. Sort by degree in complement (descending).
        # 3. Greedy color them.
        # 4. Return the max color used.
        
        # However, doing this at every node is expensive.
        # Given N=256, and time limit, we can afford a simpler bound or a limited B&B.
        
        # Alternative: Use a simple bound: |P|
        # This is weak.
        
        # Let's try a different approach:
        # Since N is small (256), we can use a recursive search with a better heuristic.
        # Or, we can use the "Max Clique" formulation and use a known efficient implementation pattern.
        
        # Let's implement a basic B&B for MIS using the bound:
        # UB = max( |P|, sum of degrees? No. )
        
        # Actually, there is a simple bound for MIS:
        # The size of the MIS is at most the number of vertices in P.
        # We can improve this by finding a clique in the complement (which is an independent set in G? No, clique in complement is independent set in G).
        # Wait, I'm confusing myself.
        
        # Let's stick to the current local search but improve the perturbation and add a "repair" step.
        # The current code scores 1.023.
        # The prompt suggests B&B with Lovasz bound. Implementing a full Lovasz bound (SDP) is too complex/slow in Python.
        # Tomita's coloring bound for Max Clique (MIS) is the standard "strong" bound.
        
        # Let's implement Tomita-style Max Clique on the Complement Graph.
        
        # Precompute Complement Adjacency?
        # comp_adj[v] = cand_mask & ~adj[v] & ~(1<<v)
        
        comp_adj = [0] * N
        for v in range(N):
            if forbidden[v]:
                comp_adj[v] = 0
            else:
                # Neighbors in complement are those in cand_mask, not v, and not in adj[v]
                comp_adj[v] = cand_mask & ~adj[v] & ~(1 << v)
        
        def max_clique_search(P, best_size, best_set):
            # P is a bitmask of candidate vertices for the clique (in comp graph)
            # We want to find a clique in comp graph (which is an independent set in original)
            
            # Bound: Greedy coloring of P in comp graph
            # We need to color the vertices in P.
            # If max_color + current_depth <= best_size, prune.
            
            # To speed up, we can just use |P| as a bound initially, or a simple greedy coloring.
            
            # Greedy Coloring for Bound:
            # 1. Get list of vertices in P.
            # 2. Sort by degree in comp graph (within P).
            # 3. Assign colors.
            
            # This is O(N^2) per node. With N=256, it might be too slow for deep search.
            # But we can limit the depth or use a simpler bound.
            
            # Let's try a simpler heuristic search first, as B&B might be too slow in Python.
            # We'll stick to the local search but improve it.
            pass

    # Let's focus on improving the Local Search.
    # The current local search only adds vertices.
    # We need to allow swaps.
    
    def local_search_with_swaps(mask, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            iters += 1
            
            # 1. Try to add
            mask = local_improve(mask)
            
            # 2. Try 1-1 swaps
            m = mask
            swapped = False
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u
                
                # Find v in comp_adj[u] (i.e., not neighbor in original)
                # that is not in mask, and replacing u with v allows adding more?
                # Simple 1-1 swap: remove u, add v. Size same.
                # Only useful if it leads to a better local optimum.
                
                # Find v in candidates, v not in mask, v not neighbor of (mask \ {u})
                # i.e., (adj[v] & (mask ^ lsb_u)) == 0
                
                # Candidates for v:
                # Must be in cand_mask
                # Must not be in mask
                # Must not be adjacent to any node in mask except possibly u
                # So (adj[v] & mask) is a subset of {u}
                # i.e., (adj[v] & mask) == 0 or (adj[v] & mask) == lsb_u
                
                # Let's iterate through all candidates not in mask.
                # This is O(N) per u, so O(N^2) per iteration.
                # N=256, N^2=65536. 1000 iters -> 65M ops. Might be tight in 58s.
                
                # Optimization:
                # Only consider v that are "close" to u?
                # Or just iterate through all v not in mask.
                
                not_in_mask = cand_mask & ~mask
                mv = not_in_mask
                while mv:
                    lsb_v = mv & -mv
                    v = lsb_v.bit_length() - 1
                    mv ^= lsb_v
                    
                    # Check if v can replace u
                    # New set would be (mask ^ lsb_u) | lsb_v
                    # We need (adj[v] & (mask ^ lsb_u)) == 0
                    if (adj[v] & (mask ^ lsb_u)) == 0:
                        # Swap is valid.
                        # Does it help?
                        # Try it and see if we can add more.
                        new_mask = (mask ^ lsb_u) | lsb_v
                        new_mask = local_improve(new_mask)
                        if new_mask.bit_count() > mask.bit_count():
                            mask = new_mask
                            improved = True
                            swapped = True
                            break
                        # Even if size doesn't increase immediately, it might escape a local optimum.
                        # But for now, let's only accept if size increases.
                
                if swapped:
                    break
                    
        return mask

    # Main Loop
    best_mask = 0
    best_size = 0
    
    # Initial greedy
    mask = greedy_init()
    mask = local_improve(mask)
    best_mask = mask
    best_size = mask.bit_count()
    
    # ILS
    while time.time() - start_time < DEADLINE:
        current = best_mask
        
        # Perturb: remove random vertices
        # Number of removals proportional to size?
        num_removals = random.randint(2, 5)
        for _ in range(num_removals):
            if not current:
                break
            m_iter = current
            # Pick random set bit
            count = current.bit_count()
            r = random.randint(1, count)
            # Find r-th set bit
            # Simple way: iterate
            cnt = 0
            m_temp = current
            while m_temp:
                lsb = m_temp & -m_temp
                cnt += 1
                if cnt == r:
                    current ^= lsb
                    break
                m_temp ^= lsb
        
        # Improve
        current = local_improve(current)
        current = local_search_with_swaps(current, max_iters=50)
        
        if current.bit_count() > best_size:
            best_size = current.bit_count()
            best_mask = current
        
        # Also try fresh greedy with randomization
        if random.random() < 0.3:
            def greedy_random():
                available_mask = cand_mask
                selected_mask = 0
                while available_mask:
                    best_v = -1
                    min_deg = N + 1
                    m = available_mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        deg = (adj[v] & available_mask).bit_count()
                        deg += random.uniform(0, 1.0)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        if deg == 0:
                            break
                    if best_v == -1:
                        break
                    selected_mask |= (1 << best_v)
                    available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
                return selected_mask

            mask_r = greedy_random()
            mask_r = local_improve(mask_r)
            mask_r = local_search_with_swaps(mask_r, max_iters=50)
            if mask_r.bit_count() > best_size:
                best_size = mask_r.bit_count()
                best_mask = mask_r

    # Convert best_mask to list of values (1-based)
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
