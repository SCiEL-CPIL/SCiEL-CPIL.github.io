import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    def has_4_collinear(points):
        n = len(points)
        # O(N^3) check
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False

    def max_clique_size(points):
        n = len(points)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                safe = True
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            safe = False
                            break
                if safe:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_size = 0
        
        def expand(P, depth):
            nonlocal best_size
            if not P:
                if depth > best_size:
                    best_size = depth
                return
            
            if depth + bin(P).count('1') <= best_size:
                return

            # Pick a vertex
            # Heuristic: pick vertex with max degree in P?
            # For N=20, simple pick is fine.
            # Let's try to pick a vertex that is part of many potential cliques?
            # Just pick lowest bit for determinism/speed.
            u_bit = P & -P
            u = u_bit.bit_length() - 1
            
            # Branch Include
            new_P = (P ^ u_bit) & adj[u]
            expand(new_P, depth + 1)
            
            # Branch Exclude
            expand(P ^ u_bit, depth)
            
        expand((1 << n) - 1, 0)
        return best_size

    def local_search(initial_config):
        best_cfg = [p[:] for p in initial_config]
        best_score = max_clique_size(best_cfg)
        
        no_improve_count = 0
        while time.time() < deadline:
            improved = False
            # Shuffle point order to avoid bias
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline: break
                x, y = best_cfg[i]
                # Try a few random moves
                # Radius depends on no_improve_count?
                radius = 5 + no_improve_count // 5
                for _ in range(20):
                    nx = random.randint(x - radius, x + radius)
                    ny = random.randint(y - radius, y + radius)
                    if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                        continue
                    if (nx, ny) == (x, y):
                        continue
                    
                    new_cfg = best_cfg[:]
                    new_cfg[i] = [nx, ny]
                    
                    if len(set(map(tuple, new_cfg))) < 20:
                        continue
                    
                    if has_4_collinear(new_cfg):
                        continue
                    
                    score = max_clique_size(new_cfg)
                    
                    if score < best_score:
                        best_cfg = new_cfg
                        best_score = score
                        improved = True
                        break
                    elif score == best_score:
                        # Accept with small probability to explore
                        if random.random() < 0.1:
                            best_cfg = new_cfg
                            # best_score stays same
                            improved = True
                            break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
                
        return best_cfg, best_score

    # Generate initial configurations
    inits = []
    
    # Grid 1
    g1 = [[i, j] for i in range(5) for j in range(4)]
    # Perturb
    for _ in range(100):
        p = g1[:]
        valid = True
        for i in range(20):
            if random.random() < 0.5:
                p[i][0] += random.choice([-1, 1])
                p[i][1] += random.choice([-1, 1])
            if len(set(map(tuple, p))) < 20 or has_4_collinear(p):
                valid = False
                break
        if valid:
            inits.append(p)
            if len(inits) >= 3: break
            
    if not inits:
        # Fallback valid config
        inits.append([[x, x*x] for x in range(-10, 10)])

    best_global_cfg = None
    best_global_score = 20
    
    for init in inits:
        if time.time() > deadline - 10: break
        cfg, score = local_search(init)
        if score < best_global_score:
            best_global_score = score
            best_global_cfg = cfg
            
    if best_global_cfg is None:
        return inits[0]
        
    return best_global_cfg
