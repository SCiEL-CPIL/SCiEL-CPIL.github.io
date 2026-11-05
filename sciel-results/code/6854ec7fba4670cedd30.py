import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    index_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in range(4):
            index_to_aps[ap[pos]].append((idx, pos))
            
    def solve(k, time_limit):
        colors = [-1] * n
        start_k = time.time()
        
        def dfs(idx):
            if time.time() > start_k + time_limit:
                return False
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            if idx == 0:
                candidates = [0]
            elif idx == 1:
                candidates = [0, 1]
            else:
                candidates = range(k)
                
            for c in candidates:
                valid = True
                for ap_idx, pos in index_to_aps[idx]:
                    ap = aps[ap_idx]
                    
                    assigned = 0
                    distinct = set()
                    for p in range(4):
                        if p == pos:
                            distinct.add(c)
                            assigned += 1
                        else:
                            col = colors[ap[p]]
                            if col != -1:
                                distinct.add(col)
                                assigned += 1
                            
                    if assigned == 4:
                        if len(distinct) < 3:
                            valid = False
                            break
                            
                if valid:
                    colors[idx] = c
                    if dfs(idx + 1):
                        return True
                    colors[idx] = -1
            return False
            
        return dfs(0)

    # Try K=3 with 30s, K=4 with 30s, etc.
    # Total 85s.
    time_per_k = 20
    for k in range(3, 129):
        if time.time() > deadline - time_per_k:
            break
        if solve(k, time_per_k):
            return colors
            
    return list(range(n))

# EVOLVE-BLOCK-END
