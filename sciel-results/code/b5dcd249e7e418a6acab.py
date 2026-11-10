import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Directly search for 3 edge-disjoint Hamiltonian cycles
    # The graph is 6-regular, so 3 is the theoretical max.
    cycles = _search_for_cycles(adj, n)
    
    return [[int(v) for v in c] for c in cycles]


def _search_for_cycles(adj, n):
    """Systematic search for 3 edge-disjoint Hamiltonian cycles."""
    start_time = time.time()
    limit = 50.0  # Safety margin
    
    def find_ham_cycle(current_adj, start, rng):
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack(current, depth):
            if depth == n:
                return start in current_adj[current]
            
            neighbors = [v for v in current_adj[current] if not visited[v]]
            if not neighbors:
                return False
            
            # Warnsdorff-like heuristic
            def score(v):
                cnt = sum(1 for w in current_adj[v] if not visited[w])
                if depth == n - 1:
                    if start not in current_adj[v]:
                        return float('inf')
                    return -1
                return cnt + rng.random() * 0.1
            
            neighbors.sort(key=score)
            
            for v in neighbors:
                if depth == n - 1 and start not in current_adj[v]:
                    continue
                path.append(v)
                visited[v] = True
                if backtrack(v, depth + 1):
                    return True
                path.pop()
                visited[v] = False
            return False
        
        if backtrack(start, 1):
            return path
        return None
    
    best = []
    # Try deterministic seeds first for reliability
    seeds = [0, 1, 2, 3, 4] + list(range(5, 500))
    for seed in seeds:
        if time.time() - start_time > limit:
            break
        rng = random.Random(seed)
        cur_adj = [set(s) for s in adj]
        cycles = []
        for _ in range(3):
            found = None
            for s in range(n):
                if not cur_adj[s]:
                    continue
                c = find_ham_cycle(cur_adj, s, rng)
                if c:
                    found = c
                    break
            if not found:
                break
            cycles.append(found)
            for i in range(len(found)):
                u, v = found[i], found[(i+1) % len(found)]
                cur_adj[u].discard(v)
                cur_adj[v].discard(u)
        if len(cycles) > len(best):
            best = cycles
            if len(best) == 3:
                break
    
    return best
# EVOLVE-BLOCK-END
