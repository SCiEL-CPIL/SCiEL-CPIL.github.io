"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import random
    from collections import defaultdict

    N = 128
    numbers = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # Precompute bad triples: (a, b, c) with a < b < c and gcd(a,b)=gcd(a,c)=gcd(b,c)
    # Also build conflict structure: for each pair (x,y), which z make (x,y,z) bad
    # And for each vertex, the set of vertices it conflicts with in some triple
    bad_triples = []
    # For each vertex v, store the set of other vertices u such that there exists w
    # with (v, u, w) being a bad triple. This helps with quick conflict checks.
    # Actually, let's store: for each pair (x, y), the set of z such that (x,y,z) is bad
    pair_conflicts = defaultdict(set)  # (min(x,y), max(x,y)) -> set of z
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d_ab = gcd_matrix[a][b]
            for c in range(b + 1, N + 1):
                if gcd_matrix[a][c] == d_ab and gcd_matrix[b][c] == d_ab:
                    bad_triples.append((a, b, c))
                    # Add to pair conflicts
                    pair_conflicts[(a, b)].add(c)
                    pair_conflicts[(a, c)].add(b)
                    pair_conflicts[(b, c)].add(a)

    # Compute degree for each vertex
    degree = [0] * (N + 1)
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    def is_valid_addition(candidate, current_set, gcd_matrix):
        """Check if candidate can be added to current_set without forming a bad triple."""
        cs = list(current_set)
        n_cs = len(cs)
        for i in range(n_cs):
            u = cs[i]
            g_cu = gcd_matrix[candidate][u]
            for j in range(i + 1, n_cs):
                v = cs[j]
                g_cv = gcd_matrix[candidate][v]
                if g_cu == g_cv:
                    if gcd_matrix[u][v] == g_cu:
                        return False
        return True

    def is_valid_excluding(candidate, current_set, excluded, gcd_matrix):
        """Check if candidate can be added to current_set minus excluded."""
        cs = [x for x in current_set if x != excluded]
        n_cs = len(cs)
        for i in range(n_cs):
            u = cs[i]
            g_cu = gcd_matrix[candidate][u]
            for j in range(i + 1, n_cs):
                v = cs[j]
                g_cv = gcd_matrix[candidate][v]
                if g_cu == g_cv:
                    if gcd_matrix[u][v] == g_cu:
                        return False
        return True

    def greedy_construct(seed):
        """Greedy construction with randomized tie-breaking."""
        rng = random.Random(seed)
        # Create list of vertices with degree
        vertices = list(range(1, N + 1))
        # Shuffle with bias toward lower degree: sort by (degree + random_noise, vertex)
        # Use a small random perturbation to break ties
        scored = [(degree[v] + rng.random() * 0.5, v) for v in vertices]
        scored.sort()
        
        S = set()
        for _, v in scored:
            if is_valid_addition(v, S, gcd_matrix):
                S.add(v)
        return S

    def local_search(S, max_iters=50):
        """Local search with 1-for-k exchanges and additions."""
        S = set(S)
        
        for _ in range(max_iters):
            improved = False
            
            # Step 1: Try to add any valid vertex
            for v in range(1, N + 1):
                if v not in S:
                    if is_valid_addition(v, S, gcd_matrix):
                        S.add(v)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Step 2: Try 1-for-2 exchange
            # For each u in S, remove u and see how many vertices can be added
            best_u = None
            best_new_S = None
            best_gain = 0
            
            for u in list(S):
                S_u = S - {u}
                # Find all vertices that can be added to S_u
                addable = []
                for v in range(1, N + 1):
                    if v not in S_u and v != u:
                        if is_valid_addition(v, S_u, gcd_matrix):
                            addable.append(v)
                
                if len(addable) >= 2:
                    # Greedily add as many as possible
                    S_greedy = set(S_u)
                    count_added = 0
                    for v in addable:
                        if v not in S_greedy:
                            if is_valid_addition(v, S_greedy, gcd_matrix):
                                S_greedy.add(v)
                                count_added += 1
                    
                    if count_added >= 2:
                        gain = count_added - 1  # Net gain
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                            best_new_S = S_greedy
            
            if best_u is not None:
                S = best_new_S
                improved = True
                continue
            
            # Step 3: Try 1-for-1 swap to potentially enable future improvements
            # Swap high-degree u with low-degree v if valid
            swapped = False
            for u in sorted(S, key=lambda x: -degree[x]):
                for v in sorted(range(1, N + 1), key=lambda x: degree[x]):
                    if v in S:
                        continue
                    if degree[v] >= degree[u]:
                        continue
                    if is_valid_excluding(v, S, u, gcd_matrix):
                        S.remove(u)
                        S.add(v)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                break
        
        return S

    # Run multiple randomized restarts
    best_S = set()
    best_size = 0
    num_restarts = 30
    
    for i in range(num_restarts):
        S = greedy_construct(seed=i)
        S = local_search(S, max_iters=30)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
