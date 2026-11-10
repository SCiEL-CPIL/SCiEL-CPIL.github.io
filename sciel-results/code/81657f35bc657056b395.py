import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    
    n = 15
    
    def is_valid(adj, degs):
        """Check constraint: no path s-u-v-t with deg(s)==deg(t)."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # s in N(u)\{v}, t in N(v)\{u}, s!=t
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if degs[s] == degs[t]:
                            return False
        return True
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    best_edges = None
    best_count = 0
    
    random.seed(12345)
    
    for attempt in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial graph
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    adj[i].add(j)
                    adj[j].add(i)
        
        degs = [len(adj[i]) for i in range(n)]
        current_count = edge_count(adj)
        
        if not is_valid(adj, degs):
            # Try to fix by removing problematic edges
            for _ in range(100):
                valid = True
                for u in range(n):
                    for v in adj[u]:
                        if v <= u:
                            continue
                        for s in adj[u]:
                            if s == v:
                                continue
                            for t in adj[v]:
                                if t == u or t == s:
                                    continue
                                if degs[s] == degs[t]:
                                    # Remove edge (u,v)
                                    adj[u].discard(v)
                                    adj[v].discard(u)
                                    degs[u] -= 1
                                    degs[v] -= 1
                                    current_count -= 1
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    break
        
        if not is_valid(adj, degs):
            continue
        
        # Local search: add edges
        for step in range(1000):
            if time.time() > deadline - 10.0:
                break
            
            added = False
            # Try all missing edges
            for i in range(n):
                for j in range(i+1, n):
                    if j in adj[i]:
                        continue
                    adj[i].add(j)
                    adj[j].add(i)
                    degs[i] += 1
                    degs[j] += 1
                    if is_valid(adj, degs):
                        added = True
                        current_count += 1
                        break
                    else:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        degs[i] -= 1
                        degs[j] -= 1
                if added:
                    break
            
            if not added:
                break
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [(i, j) for i in range(n) for j in adj[i] if j > i]
    
    if best_edges is None:
        best_edges = [(0, v) for v in range(1, n)]
        best_count = n - 1
    
    result = [[int(e[0]), int(e[1])] for e in best_edges]
    return result
