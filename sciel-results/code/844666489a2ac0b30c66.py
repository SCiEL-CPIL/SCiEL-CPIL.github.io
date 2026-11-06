import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def edge_index(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    def has_octahedron(colors, n):
        if n < 6:
            return False
        adj = [0] * (2 * n)
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index(i, j, n)]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
        
        for color in range(2):
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[color * n + a] >> b) & 1):
                        continue
                    for c in range(b + 1, n):
                        if not ((adj[color * n + a] >> c) & 1) or not ((adj[color * n + b] >> c) & 1):
                            continue
                        for d in range(c + 1, n):
                            if not ((adj[color * n + a] >> d) & 1) or not ((adj[color * n + b] >> d) & 1) or not ((adj[color * n + c] >> d) & 1):
                                continue
                            for e in range(d + 1, n):
                                if not ((adj[color * n + a] >> e) & 1) or not ((adj[color * n + b] >> e) & 1) or not ((adj[color * n + c] >> e) & 1) or not ((adj[color * n + d] >> e) & 1):
                                    continue
                                for f in range(e + 1, n):
                                    if ((adj[color * n + a] >> f) & 1) and ((adj[color * n + b] >> f) & 1) and ((adj[color * n + c] >> f) & 1) and ((adj[color * n + d] >> f) & 1) and ((adj[color * n + e] >> f) & 1):
                                        return True
        return False
    
    # Start with Paley n=17
    best_n = 17
    best_colors = [0 if (j - i) % 17 in {x*x % 17 for x in range(1,17)} else 1 for i in range(17) for j in range(i+1, 17)]
    
    # Try to extend to larger n using random local search
    for target_n in range(18, 31):
        if time.time() > deadline - 10:
            break
        found = False
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            colors = [random.randint(0, 1) for _ in range(target_n*(target_n-1)//2)]
            for it in range(1000):
                if time.time() > deadline - 10:
                    break
                if not has_octahedron(colors, target_n):
                    found = True
                    break
                i = random.randint(0, target_n-1)
                j = random.randint(i+1, target_n-1)
                idx = edge_index(i, j, target_n)
                colors[idx] ^= 1
            if found:
                best_n = target_n
                best_colors = colors
                break
        if not found:
            break
    
    return {'n': best_n, 'colors': best_colors}
