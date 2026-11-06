import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    for n in range(47, 38, -1):
        target_min_deg = max(0, n - 39)
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            steps = 0
            max_steps = 1000
            while steps < max_steps and time.time() < deadline:
                steps += 1
                # Check for C4s
                has_c4 = False
                c4_info = None
                for u in range(n):
                    nu = adj[u]
                    for v in range(u+1, n):
                        is_adj = (nu >> v) & 1
                        common = nu & adj[v]
                        w1 = -1
                        w2 = -1
                        temp = common
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp ^= lsb
                            if w1 == -1:
                                w1 = w
                            else:
                                w2 = w
                                break
                                
                        if w2 != -1:
                            if is_adj:
                                has_c4 = True
                                c4_info = (u, v, w1, w2)
                                break
                            else:
                                if (adj[w1] >> w2) & 1:
                                    has_c4 = True
                                    c4_info = (u, v, w1, w2)
                                    break
                    if has_c4:
                        break
                
                if has_c4:
                    u, v, w1, w2 = c4_info
                    if (adj[u] >> v) & 1:
                        edges_to_break = [(u, w1), (w1, v), (v, w2), (w2, u)]
                    else:
                        edges_to_break = [(u, w1), (w1, w2), (w2, v), (v, u)]
                    
                    a, b = random.choice(edges_to_break)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    min_deg = min(bin(adj[i]).count("1") for i in range(n))
                    if min_deg >= target_min_deg:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        low_v = min(range(n), key=lambda i: bin(adj[i]).count("1"))
                        candidates = [u for u in range(n) if u != low_v and not ((adj[low_v] >> u) & 1)]
                        if candidates:
                            u = random.choice(candidates)
                            adj[low_v] |= (1 << u)
                            adj[u] |= (1 << low_v)
                        else:
                            # Complete graph? But C4-free complete graph is K3.
                            # If n>3, it's not complete.
                            # If no candidates, it means low_v is connected to everyone.
                            # This implies high degree.
                            # But we are in "else" of min_deg >= target, so degree is low?
                            # Contradiction unless n is small.
                            pass

    return {'n': 39, 'edges': []}
