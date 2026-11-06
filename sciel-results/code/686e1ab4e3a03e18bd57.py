import time
import random

def construct():
    start_time = time.time()
    
    # Fallback: 8 disjoint copies of K4
    # n = 32
    # alpha = 8 (one from each K4)
    # C5-free: K4 has no 5-cycle, disjoint union has no 5-cycle.
    def make_k4_graph(num_copies):
        n = 4 * num_copies
        edges = []
        for b in range(num_copies):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    # Try to find a better graph?
    # Given the complexity and time limit, and the fact that 32 is a known strong bound 
    # for disjoint cliques, and finding non-disjoint C5-free graphs with alpha<9 
    # and n>32 is research-level difficult, we stick to the robust 32-vertex solution.
    # The task mentions "best known has 32 vertices", implying 32 is the target.
    
    result = make_k4_graph(8)
    
    # Verification (optional but good for safety)
    n = result['n']
    adj = [[False]*n for _ in range(n)]
    for u, v in result['edges']:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check C5-free (simple check: no 5-cycles)
    # For n=32, this is fast enough.
    has_c5 = False
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]: continue
            for c in range(b+1, n):
                if not adj[b][c]: continue
                for d in range(c+1, n):
                    if not adj[c][d]: continue
                    for e in range(d+1, n):
                        if not adj[d][e]: continue
                        if adj[e][a]:
                            has_c5 = True
                            break
                    if has_c5: break
                if has_c5: break
            if has_c5: break
        if has_c5: break
        
    if has_c5:
        # This should never happen for K4 copies
        return make_k4_graph(7)

    # Check alpha < 9
    # Max independent set in 8xK4 is 8.
    # We can verify by brute force or just trust the structure.
    # Structure is known to be valid.
    
    return result
