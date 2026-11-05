import time
import random
import itertools

DEADLINE = 115.0

def has_c5(adj, n):
    """Check if graph contains a 5-cycle."""
    for v in range(n):
        neighbors = [u for u in range(n) if adj[v][u]]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                d = neighbors[j]
                # Find b, c such that a-b-c-d is a path
                # b, c distinct, not v, a, d
                for b in range(n):
                    if b == v or b == a or b == d:
                        continue
                    if not adj[a][b]:
                        continue
                    for c in range(n):
                        if c == v or c == a or c == d or c == b:
                            continue
                        if not adj[d][c]:
                            continue
                        if not adj[b][c]:
                            continue
                        return True
    return False

def has_independent_set(adj, n, k=9):
    """Check for independent set of size k."""
    # Greedy + Backtracking
    remaining = set(range(n))
    chosen = []
    while remaining and len(chosen) < k:
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in remaining if adj[v][u])
            if deg < best_deg:
                best_deg = deg
                best_v = v
        chosen.append(best_v)
        remaining.discard(best_v)
        remaining -= {u for u in remaining if adj[best_v][u]}
    
    if len(chosen) >= k:
        return True
    
    # Backtracking for exact check if n is small
    def backtrack(remaining, chosen_set):
        if len(chosen_set) >= k:
            return True
        if not remaining:
            return False
        if len(chosen_set) + len(remaining) < k:
            return False
        v = min(remaining, key=lambda x: sum(1 for u in remaining if adj[x][u]))
        new_remaining = remaining - {v} - {u for u in remaining if adj[v][u]}
        if backtrack(new_remaining, chosen_set | {v}):
            return True
        if backtrack(remaining - {v}, chosen_set):
            return True
        return False
    
    if n <= 35:
        return backtrack(set(range(n)), set())
    return False

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Base construction: 8 disjoint K4s (n=32)
    n_base = 32
    adj_base = [[False] * n_base for _ in range(n_base)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(4):
                if i != j:
                    adj_base[base + i][base + j] = True
    
    best_n = n_base
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33
    # Strategy: Randomly perturb the base or try specific additions.
    # Given the proof that simple addition to 8 K4s fails, we try a randomized search
    # on 33 vertices directly, starting from a random graph and removing C5s.
    
    random.seed(42)
    
    # Attempt to construct n=33
    found_33 = False
    adj_33 = None
    
    # Time budget for search: 100s
    search_deadline = start_time + 100.0
    
    while time.time() < search_deadline and not found_33:
        # Generate a random graph on 33 vertices
        # Bias towards dense graphs to keep alpha low
        adj_try = [[False] * 33 for _ in range(33)]
        for i in range(33):
            for j in range(i + 1, 33):
                if random.random() < 0.6:
                    adj_try[i][j] = True
                    adj_try[j][i] = True
        
        # Remove edges to eliminate C5
        # Iterate: find a C5, remove a random edge from it
        iterations = 0
        while iterations < 1000:
            if time.time() > search_deadline:
                break
            # Check for C5
            c5_found = False
            # Efficient C5 check and extraction
            for v in range(33):
                neighbors = [u for u in range(33) if adj_try[v][u]]
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        a = neighbors[i]
                        d = neighbors[j]
                        for b in range(33):
                            if b == v or b == a or b == d:
                                continue
                            if not adj_try[a][b]:
                                continue
                            for c in range(33):
                                if c == v or c == a or c == d or c == b:
                                    continue
                                if not adj_try[d][c]:
                                    continue
                                if not adj_try[b][c]:
                                    # Found C5: v-a-b-c-d-v
                                    # Remove a random edge from this cycle
                                    edges_to_remove = [(v,a), (a,b), (b,c), (c,d), (d,v)]
                                    u, w = random.choice(edges_to_remove)
                                    adj_try[u][w] = False
                                    adj_try[w][u] = False
                                    c5_found = True
                                    break
                            if c5_found:
                                break
                        if c5_found:
                            break
                    if c5_found:
                        break
                if c5_found:
                    break
            if not c5_found:
                # Check independent set
                if not has_independent_set(adj_try, 33, 9):
                    found_33 = True
                    adj_33 = adj_try
                    break
            iterations += 1
    
    if found_33:
        best_n = 33
        best_edges = []
        for i in range(33):
            for j in range(i + 1, 33):
                if adj_33[i][j]:
                    best_edges.append([i, j])
    else:
        # Fallback to 32
        best_n = 32
        best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
        
    return {'n': best_n, 'edges': best_edges}
