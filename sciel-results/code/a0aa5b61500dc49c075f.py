import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_graph = {'n': n_seed, 'edges': edges_seed}
    
    # Try n=30
    # Random search for circulant
    # Check triangle-free and alpha < 10
    
    def is_valid_circulant(n, S):
        S_set = set(S)
        # Triangle free
        for a in S:
            for b in S:
                if a != b and (a - b) % n in S_set:
                    return False
        
        # Alpha < 10
        # Use a simple greedy + local search to find large independent set
        # If we find one of size 10, invalid.
        # If not, likely valid.
        
        # Heuristic: Try to find independent set of size 10
        for _ in range(500):
            # Random subset of size 10
            verts = random.sample(range(n), 10)
            indep = True
            for u in verts:
                for v in verts:
                    if u == v: continue
                    if (v - u) % n in S_set:
                        indep = False
                        break
                if not indep: break
            if indep:
                return False
        return True

    for n in [30, 29, 28]:
        if time.time() > deadline:
            break
        for _ in range(10000):
            if time.time() > deadline:
                break
            S = [i for i in range(1, n // 2 + 1) if random.random() < 0.3]
            if not S: continue
            if is_valid_circulant(n, S):
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return best_graph
