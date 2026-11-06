import time
import random

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Time allocation per n
    # n=46: 50s, n=45: 30s, n=44: 15s, n=43: 10s, n=42: 5s, n=41: 3s, n=40: 2s
    time_limits = {
        46: 50,
        45: 30,
        44: 15,
        43: 10,
        42: 5,
        41: 3,
        40: 2
    }

    for n in range(46, 39, -1):
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
            
        limit = time_limits.get(n, 1)
        n_start = time.time()
        n_deadline = n_start + limit
        
        # If we are already close to global deadline, reduce limit
        if time.time() > deadline - 5:
            break
            
        max_trials = 5000 # High number, time limited
        
        for _ in range(max_trials):
            if time.time() > n_deadline or time.time() > deadline:
                break
                
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            success = True
            steps = 0
            max_steps = n * min_deg + 100
            
            while steps < max_steps:
                if time.time() > n_deadline or time.time() > deadline:
                    success = False
                    break
                    
                min_d = min(degrees)
                if min_d >= min_deg:
                    break
                
                min_vertices = [v for v in range(n) if degrees[v] == min_d]
                u = random.choice(min_vertices)
                
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if (adj[u] >> v) & 1:
                        continue
                    candidates.append(v)
                    
                if not candidates:
                    success = False
                    break
                    
                # Sort by degree, with random tie-breaker
                candidates.sort(key=lambda x: (degrees[x], random.random()))
                
                added = False
                for v in candidates:
                    common = adj[u] & adj[v]
                    if _popcount(common) < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added = True
                        break
                
                if not added:
                    success = False
                    break
                    
                steps += 1
            
            if success and min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}

    return {'n': 39, 'edges': []}
