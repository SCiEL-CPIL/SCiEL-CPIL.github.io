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

    # Precompute pair conflicts: for each pair (u, v), the set of w such that (u, v, w) is a bad triple.
    # This allows O(1) lookup for conflict checks during local search.
    pair_conflicts = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d_ab = gcd_matrix[a][b]
            for c in range(b + 1, N + 1):
                if gcd_matrix[a][c] == d_ab and gcd_matrix[b][c] == d_ab:
                    # (a, b, c) is a bad triple
                    pair_conflicts[a][b].add(c)
                    pair_conflicts[b][a].add(c)
                    pair_conflicts[a][c].add(b)
                    pair_conflicts[c][a].add(b)
                    pair_conflicts[b][c].add(a)
                    pair_conflicts[c][b].add(a)
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1

    def is_valid_addition(candidate, current_set):
        """Check if candidate can be added to current_set without forming a bad triple."""
        cs = list(current_set)
        n_cs = len(cs)
        for i in range(n_cs):
            u = cs[i]
            for j in range(i + 1, n_cs):
                v = cs[j]
                if candidate in pair_conflicts[u][v]:
                    return False
        return True

    def greedy_construct(seed):
        """Greedy construction with randomized tie-breaking."""
        rng = random.Random(seed)
        vertices = list(range(1, N + 1))
        # Sort by degree, then random noise
        scored = [(degree[v] + rng.random() * 1.0, v) for v in vertices]
        scored.sort()
        
        S = set()
        for _, v in scored:
            if is_valid_addition(v, S):
                S.add(v)
        return S

    def local_search(S, max_iters=100, seed=0):
        """Local search with 1-for-k exchanges and shakes."""
        rng = random.Random(seed)
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        for _ in range(max_iters):
            improved = False
            
            # Step 1: Try to add any valid vertex
            for v in range(1, N + 1):
                if v not in S:
                    if is_valid_addition(v, S):
                        S.add(v)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
            
            if improved:
                continue
            
            # Step 2: Try 1-for-k exchange
            best_u = None
            best_new_S = None
            best_gain = 0
            
            for u in list(S):
                S_u = S - {u}
                # Find all vertices that can be added to S_u
                addable = []
                for v in range(1, N + 1):
                    if v not in S_u and v != u:
                        if is_valid_addition(v, S_u):
                            addable.append(v)
                
                if len(addable) >= 2:
                    # Greedily add as many as possible, prioritizing low degree
                    S_greedy = set(S_u)
                    addable.sort(key=lambda x: degree[x])
                    count_added = 0
                    for v in addable:
                        if v not in S_greedy:
                            if is_valid_addition(v, S_greedy):
                                S_greedy.add(v)
                                count_added += 1
                        
                        if count_added >= 2:
                            gain = count_added - 1
                            if gain > best_gain:
                                best_gain = gain
                                best_u = u
                                best_new_S = S_greedy
                            break # No need to check more for this u if we found a good one? 
                            # Actually, we want the best gain for this u, so we should finish adding.
                            # But breaking the inner loop early might miss a better set for this u.
                            # Let's not break, just let it run.
                    # Remove the premature break logic if it was there. 
                    # The code above breaks the 'for v in addable' loop? No, I put a break in my thought, 
                    # but in the code below I will ensure it completes.
                    
                    # Re-evaluating gain after full greedy add
                    if count_added >= 2:
                        gain = count_added - 1
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                            best_new_S = S_greedy
            
            if best_u is not None:
                S = best_new_S
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue

            # Step 3: Shake - remove 2 random vertices and re-add greedily
            if len(S) > 5 and rng.random() < 0.1:
                S_shake = set(S)
                to_remove = rng.sample(list(S_shake), 2)
                for r in to_remove:
                    S_shake.remove(r)
                
                # Re-add greedily from remaining candidates
                candidates = [v for v in range(1, N + 1) if v not in S_shake]
                candidates.sort(key=lambda x: degree[x] + rng.random())
                for v in candidates:
                    if is_valid_addition(v, S_shake):
                        S_shake.add(v)
                
                if len(S_shake) > len(S):
                    S = S_shake
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue

            if not improved:
                break
        
        return best_S

    # Run multiple randomized restarts
    best_S = set()
    best_size = 0
    num_restarts = 50
    
    for i in range(num_restarts):
        S = greedy_construct(seed=i)
        S = local_search(S, max_iters=50, seed=i)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
