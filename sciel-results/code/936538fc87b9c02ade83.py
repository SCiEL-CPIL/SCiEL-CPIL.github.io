import time
import random

def construct():
    n = 512
    deadline = time.time() + 85.0
    
    # Precompute 4-APs
    # aps[x] = list of (y, z, w)
    aps = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for d in range(1, (n - 3 * a + 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            aps[a].append((b, c, e))
            aps[b].append((a, c, e))
            aps[c].append((a, b, e))
            aps[e].append((a, b, c))
            
    def run_greedy_randomized():
        in_S = [False] * (n + 1)
        U = set(range(1, n + 1))
        conflict_count = [0] * (n + 1)
        
        while U:
            available = [x for x in U if conflict_count[x] == 0]
            if not available:
                break
            
            x = random.choice(available)
            in_S[x] = True
            U.remove(x)
            
            for (y, z, w) in aps[x]:
                if y in U and in_S[z] and in_S[w]:
                    conflict_count[y] += 1
                if z in U and in_S[y] and in_S[w]:
                    conflict_count[z] += 1
                if w in U and in_S[y] and in_S[z]:
                    conflict_count[w] += 1
                    
        return [i for i in range(1, n + 1) if in_S[i]]

    best_size = 0
    best_set = []
    
    while time.time() < deadline:
        result = run_greedy_randomized()
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    return best_set
