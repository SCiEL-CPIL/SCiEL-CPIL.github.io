import time
import random
import math

def construct():
    n = 50
    deadline = time.time() + 110
    
    def idx_of(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + j - i - 1

    # Fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    rng = random.Random(42)
    
    # Try n=50
    for restart in range(5):
        if time.time() > deadline:
            break
            
        colors = [rng.randint(0, 3) for _ in range(n*(n-1)//2)]
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        conflicts = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        conflicts += 1
                        
        if conflicts == 0:
            return {'n': n, 'colors': colors}
            
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        max_steps = 200000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if conflicts == 0:
                return {'n': n, 'colors': colors}
                
            u = rng.randint(0, n-1)
            v = rng.randint(u+1, n-1)
            
            c_old = adj[u][v]
            # Only consider edges that are in a conflict to focus search?
            # If we pick a non-conflict edge, delta is likely 0 or positive.
            # Let's check if it's a conflict.
            is_conflict = False
            for w in range(n):
                if w == u or w == v: continue
                if adj[u][w] == c_old and adj[v][w] == c_old:
                    is_conflict = True
                    break
            if not is_conflict:
                continue
                
            c_new = rng.randint(0, 3)
            while c_new == c_old:
                c_new = rng.randint(0, 3)
                
            delta = 0
            for w in range(n):
                if w == u or w == v: continue
                if adj[u][w] == c_old and adj[v][w] == c_old:
                    delta -= 1
                if adj[u][w] == c_new and adj[v][w] == c_new:
                    delta += 1
                    
            if delta < 0 or rng.random() < math.exp(-delta / T):
                idx = idx_of(u, v)
                colors[idx] = c_new
                adj[u][v] = c_new
                adj[v][u] = c_new
                conflicts += delta
                
            T *= cooling
            if T < T_min:
                T = T_min
                
    return {'n': 32, 'colors': colors32}
