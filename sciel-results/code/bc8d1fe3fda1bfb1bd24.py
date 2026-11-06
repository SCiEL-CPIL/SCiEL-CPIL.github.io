import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    best_edges = 0
    best_N = [0] * n
    
    def count_edges(N):
        c = 0
        for i in range(n):
            c += N[i].bit_count()
        return c // 2

    def is_valid(N):
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = N[i] & N[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & N[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(N):
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = N[i] & N[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & N[k]
                    if common_ijk.bit_count() >= 3:
                        return (i, j, k, common_ijk)
        return None

    def remove_random_edge_from_violation(N, viol):
        i, j, k, common = viol
        common_list = [v for v in range(n) if (common >> v) & 1]
        if not common_list:
            return N
        x = random.choice(common_list)
        y = random.choice([i, j, k])
        N[x] &= ~(1 << y)
        N[y] &= ~(1 << x)
        return N

    def try_add_edge(N, u, v):
        if (N[u] >> v) & 1:
            return True
        
        for w in range(n):
            if w == u or w == v: continue
            for x in range(w + 1, n):
                if x == u or x == v: continue
                if ((N[w] >> v) & 1) and ((N[x] >> v) & 1):
                    old_common = N[u] & N[w] & N[x]
                    if old_common.bit_count() == 2:
                        return False
        
        for w in range(n):
            if w == u or w == v: continue
            for x in range(w + 1, n):
                if x == u or x == v: continue
                if ((N[w] >> u) & 1) and ((N[x] >> u) & 1):
                    old_common = N[v] & N[w] & N[x]
                    if old_common.bit_count() == 2:
                        return False
        
        return True

    def repair_and_improve(N, max_steps=2000):
        steps = 0
        while steps < max_steps:
            viol = find_violation(N)
            if viol:
                N = remove_random_edge_from_violation(N, viol)
                steps += 1
                continue
            
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((N[i] >> j) & 1):
                        non_edges.append((i, j))
            
            if not non_edges:
                break
            
            random.shuffle(non_edges)
            improved = False
            for u, v in non_edges:
                if try_add_edge(N, u, v):
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
                    improved = True
                    break
            
            if not improved:
                break
            
            steps += 1
        return N

    def perturb(N, k):
        N = N[:]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (N[i] >> j) & 1:
                    edges.append((i, j))
        
        if len(edges) < k:
            return N
        
        random.shuffle(edges)
        for u, v in edges[:k]:
            N[u] &= ~(1 << v)
            N[v] &= ~(1 << u)
        return N

    while time.time() < deadline:
        # Strategy 1: Random start
        if random.random() < 0.5 or best_edges == 0:
            p = random.uniform(0.3, 0.8)
            N = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        N[i] |= (1 << j)
                        N[j] |= (1 << i)
        else:
            # Strategy 2: Perturb best
            k = random.randint(3, 10)
            N = perturb(best_N, k)
        
        N = repair_and_improve(N)
        
        e = count_edges(N)
        if e > best_edges:
            best_edges = e
            best_N = N[:]

    # Final validation
    if not is_valid(best_N):
        while True:
            viol = find_violation(best_N)
            if not viol: break
            best_N = remove_random_edge_from_violation(best_N, viol)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_N[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
