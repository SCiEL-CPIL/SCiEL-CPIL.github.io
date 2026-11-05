import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # Generate all 35 binary vectors of length 7 with weight 4
    vectors = []
    for comb in itertools.combinations(range(7), 4):
        v = [0] * 7
        for i in comb:
            v[i] = 1
        vectors.append(tuple(v))
    
    n = len(vectors)
    
    # Build adjacency list
    # Edge if Hamming distance is 2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(1 for a, b in zip(vectors[i], vectors[j]) if a != b)
            if dist == 2:
                edges.append([i, j])
    
    # Verification
    # 1. Check for K4
    # 2. Check for independent set of size 6
    
    # Build adjacency matrix for faster checks
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Check K4
    def has_k4():
        for a in range(n):
            for b in range(a + 1, n):
                if not adj[a][b]:
                    continue
                for c in range(b + 1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c + 1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return True
        return False

    # Check Independent Set of size 6
    # Independent set in G is a clique in complement G_bar
    # G_bar has edges where dist != 2
    # We can just check for independent set directly
    def has_independent_set_6():
        # Use a simple backtracking search or brute force if n is small
        # n=35, C(35,6) is ~1.6M, feasible
        # But we can prune
        def backtrack(start, current_set):
            if len(current_set) == 6:
                return True
            # Pruning: if not enough vertices left
            if n - start < 6 - len(current_set):
                return False
            
            for i in range(start, n):
                # Check if i can be added to current_set
                # i must be non-adjacent to all in current_set
                can_add = True
                for v in current_set:
                    if adj[v][i]:
                        can_add = False
                        break
                if can_add:
                    if backtrack(i + 1, current_set + [i]):
                        return True
            return False
        
        return backtrack(0, [])

    if has_k4():
        # Fallback to Paley 17 if construction fails (should not happen)
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    if has_independent_set_6():
        # Fallback to Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    return {'n': n, 'edges': edges}
