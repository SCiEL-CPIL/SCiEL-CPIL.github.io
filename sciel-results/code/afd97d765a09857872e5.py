import time
import random

def construct():
    N = 128
    colors = [0] * (N + 1)
    
    ap_end = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 1:
                ap_end[i].append((a, d))
                
    def solve(i, limit_nodes):
        # limit_nodes is a mutable list to track count
        if limit_nodes[0] > 1000000:
            return False
        limit_nodes[0] += 1
        
        if i > N:
            return True
        
        # Shuffle colors for randomization
        cs = [0, 1, 2]
        random.shuffle(cs)
        
        for c in cs:
            colors[i] = c
            valid = True
            for a, d in ap_end[i]:
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                c4 = c
                if len({c1, c2, c3, c4}) < 3:
                    valid = False
                    break
            if valid:
                if solve(i + 1, limit_nodes):
                    return True
        colors[i] = -1
        return False

    start_time = time.time()
    # Try a few random restarts
    for attempt in range(20):
        limit_nodes = [0]
        if solve(1, limit_nodes):
            return colors[1:N+1]
        if time.time() - start_time > 50:
            break
            
    # If all fails, return a dummy (should not happen)
    return [0]*N
