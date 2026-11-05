import time
import random
import itertools

def _build_adj(n, colors):
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    return adj

def _has_k222(adj):
    n = len(adj)
    if n < 6:
        return False
    
    # Precompute partitions of 6 elements into 3 pairs
    # Indices 0..5
    partitions = [
        [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
    ]
    
    # Iterate over all 6-subsets
    # Optimization: If we find one, return True
    for combo in itertools.combinations(range(n), 6):
        # Map combo indices to 0..5
        # combo[0] -> 0, combo[1] -> 1, etc.
        
        for part in partitions:
            # Check intra-pair edges are 0
            valid = True
            for p in part:
                if adj[combo[p[0]]][combo[p[1]]] == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check inter-pair edges are 1
            for i in range(3):
                for j in range(i+1, 3):
                    for u in part[i]:
                        for v in part[j]:
                            if adj[combo[u]][combo[v]] == 0:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            
            if valid:
                return True
                
    return False

def _is_valid(n, colors):
    adj0 = _build_adj(n, colors)
    if _has_k222(adj0):
        return False
    
    # Build adj1 (complement)
    # Or just invert colors and check
    colors1 = [1 - c for c in colors]
    adj1 = _build_adj(n, colors1)
    if _has_k222(adj1):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Try Paley 29
    p = 29
    squares = {x * x % p for x in range(1, p)}
    colors = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    
    if _is_valid(p, colors):
        return {'n': p, 'colors': colors}
        
    # 2. If Paley 29 fails, try a quick random search for n=29
    # This is unlikely to be needed, but adds robustness
    # We will do a few random restarts with local search if time permits
    # But for now, if Paley 29 fails, we might just return 17
    # However, let's try one random instance of 29
    # Actually, random 29 is very likely to have K222.
    # So we stick to Paley.
    
    # 3. Fallback to n=17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    colors = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'colors': colors}
