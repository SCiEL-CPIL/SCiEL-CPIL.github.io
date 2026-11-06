"""Q3-free graph on 16 vertices - maximize edges via greedy Q3 destruction."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110
    n = 16
    
    q3_edges = []
    for i in range(8):
        for d in [1, 2, 4]:
            j = i ^ d
            if i < j:
                q3_edges.append((i, j))
    
    def is_q3_in_graph(adj, mapping):
        for a, b in q3_edges:
            ua, ub = mapping[a], mapping[b]
            if ua > ub:
                ua, ub = ub, ua
            if (ua, ub) not in adj:
                return False
        return True
    
    # Start with K16
    adj = set()
    for i in range(n):
        for j in range(i+1, n):
            adj.add((i, j))
    
    start_time = time.time()
    random.seed(42)
    
    for it in range(200000):
        if time.time() - start_time > 108:
            break
        
        found = False
        for _ in range(300):
            subset = random.sample(range(n), 8)
            perm = subset[:]
            random.shuffle(perm)
            mapping = {i: perm[i] for i in range(8)}
            if is_q3_in_graph(adj, mapping):
                found = True
                break
        
        if not found:
            continue
        
        a, b = random.choice(q3_edges)
        ua, ub = mapping[a], mapping[b]
        if ua > ub:
            ua, ub = ub, ua
        adj.discard((ua, ub))
    
    return [[u, v] for u, v in adj]
