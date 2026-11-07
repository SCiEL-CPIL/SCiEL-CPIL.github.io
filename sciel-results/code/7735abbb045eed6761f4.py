import time
import random
import math

def check_constraint_fast(adj, n):
    deg = [len(adj[i]) for i in range(n)]
    num_edges = sum(deg) // 2
    violations = 0
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        violations += 1
    return num_edges, violations

def construct():
    n = 15
    deadline = time.time() + 85
    
    best_edges = 0
    best_adj = None
    
    # Try multiple restarts
    # Each restart gets a fraction of the time
    # Or just run until deadline and keep global best
    
    # Initialize with a few different densities
    densities = [0.5, 0.7, 0.9, 0.95, 1.0]
    
    for density in densities:
        if time.time() > deadline:
            break
            
        # Time per restart: remaining time / remaining restarts
        # Simple: just run a fixed number of iterations or until time slice ends
        # Let's allocate 15s per restart? 5 restarts * 15s = 75s. Safe.
        restart_deadline = time.time() + 15
        
        adj = [set() for _ in range(n)]
        random.seed(int(time.time()*1000) + density*100)
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i].add(j)
                    adj[j].add(i)
        
        current_e, current_v = check_constraint_fast(adj, n)
        
        if current_v == 0 and current_e > best_edges:
            best_edges = current_e
            best_adj = [set(s) for s in adj]
            
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        def get_cost(e, v):
            if v == 0:
                return -e
            else:
                return 10000 + v
        
        current_cost = get_cost(current_e, current_v)
        
        while time.time() < restart_deadline:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            exists = v in adj[u]
            
            if exists:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
                
            new_e, new_v = check_constraint_fast(adj, n)
            new_cost = get_cost(new_e, new_v)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_e = new_e
                current_v = new_v
                
                if new_v == 0 and new_e > best_edges:
                    best_edges = new_e
                    best_adj = [set(s) for s in adj]
            else:
                if exists:
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
            T *= cooling
            if T < T_min:
                T = T_min

    if best_adj is None:
        return []
        
    result = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                result.append([int(i), int(j)])
                
    return result
