import random
import time

def construct():
    N = 24
    # Generate edge order
    rng = random.Random(614)
    edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    rng.shuffle(edges)
    edge_pos = {}
    for idx, (u, v) in enumerate(edges):
        edge_pos[(u, v)] = idx + 1
        edge_pos[(v, u)] = idx + 1

    # Precompute adjacency lists for quick lookup?
    # adj[u] = list of v such that (u,v) is an edge
    # But K changes, so we need to filter.
    # We can just iterate all v and check edge_pos.
    
    def has_ham_cycle(K):
        # Use DFS with memoization
        # State: (mask, last)
        # Returns: tuple of vertices in path, or None
        memo = {}
        
        def dfs(last, mask, path):
            key = (mask, last)
            if key in memo:
                return memo[key]
            
            if mask == (1 << N) - 1:
                if edge_pos[(last, 0)] <= K:
                    memo[key] = path + [0]
                    return memo[key]
                else:
                    memo[key] = None
                    return None
            
            # Try all neighbors
            # Optimization: sort neighbors by edge_pos? Or just iterate.
            for v in range(N):
                if not (mask & (1 << v)):
                    if edge_pos[(last, v)] <= K:
                        res = dfs(v, mask | (1 << v), path + [v])
                        if res:
                            memo[key] = res
                            return res
            
            memo[key] = None
            return None
        
        # Start from 0
        # Note: We fix the start at 0 to reduce symmetry.
        # A Hamiltonian cycle can always be rotated to start at 0.
        return dfs(0, 1, [0])

    # Iterate K
    best_cycle = None
    for K in range(1, 277):
        # Time check
        if time.time() > 50: # Safety margin
            break
        cycle = has_ham_cycle(K)
        if cycle:
            best_cycle = cycle
            break

    if best_cycle:
        # Return permutation without the closing 0?
        # The task says "permutation of vertices 0..23, describing a Hamilton cycle with implicit closing edge".
        # Usually this means [v0, v1, ..., v23] where edges are (v0,v1)...(v22,v23),(v23,v0).
        # My dfs returns [0, ..., 0]. I should strip the last 0.
        return best_cycle[:-1]
    else:
        # Fallback: return identity or a random valid cycle
        # A complete graph always has a Hamiltonian cycle.
        # Just return range(N)
        return list(range(N))
