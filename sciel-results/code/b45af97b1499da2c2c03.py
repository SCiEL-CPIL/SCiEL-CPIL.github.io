import math
import random

def construct():
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    def greedy_clique(start_idx, seed):
        rng = random.Random(seed)
        current = {start_idx}
        # Available vertices that connect to all in current
        avail = [i for i in range(n) if i != start_idx and all(adj[i][j] for j in current)]
        
        while avail:
            # Pick vertex with max degree within avail (or overall)
            # Score by number of connections to current + degree in avail
            def score(i):
                # Connections to current
                conn_cur = sum(1 for j in current if adj[i][j])
                # Degree among avail
                deg_avail = sum(1 for k in avail if adj[i][k])
                return (conn_cur, deg_avail, rng.random())
            
            avail.sort(key=score, reverse=True)
            best_v = avail[0]
            current.add(best_v)
            # Update avail: only keep those connected to best_v
            avail = [i for i in avail if i != best_v and adj[i][best_v]]
        
        return current
    
    best = set()
    for seed in range(100):
        clique = greedy_clique(idx_255, seed)
        if len(clique) > len(best):
            best = clique
    
    result = [candidates[i] for i in best]
    return result
