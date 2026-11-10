import time
import itertools
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    random.seed(99)
    
    q3_edges = [(u,v) for u in range(8) for v in range(u+1,8) if bin(u^v).count('1')==1]
    
    def has_q3(adj, subset):
        local = [0]*8
        for i in range(8):
            ui = subset[i]
            m = 0
            for j in range(8):
                if i != j and (adj[ui] >> subset[j] & 1):
                    m |= (1<<j)
            local[i] = m
        for i in range(8):
            if bin(local[i]).count('1') < 3:
                return False
        for perm in itertools.permutations(range(8)):
            ok = True
            for a,b in q3_edges:
                if not (local[perm[a]] >> perm[b] & 1):
                    ok = False
                    break
            if ok:
                return True
        return False
    
    # Start with K_{3,13}
    adj = [0]*n
    for i in range(3):
        for j in range(3, 16):
            adj[i] |= (1<<j)
            adj[j] |= (1<<i)
    
    def edge_list():
        return [[i,j] for i in range(n) for j in range(i+1,n) if adj[i]>>j&1]
    
    # Try to add edges one by one
    while time.time() < deadline - 5:
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline - 5:
                    break
                if not (adj[i] >> j & 1):
                    adj[i] |= (1<<j)
                    adj[j] |= (1<<i)
                    free = True
                    for subset in itertools.combinations(range(n), 8):
                        if has_q3(adj, subset):
                            free = False
                            break
                    if free:
                        added = True
                    else:
                        adj[i] &= ~(1<<j)
                        adj[j] &= ~(1<<i)
            if time.time() > deadline - 5:
                break
        if not added:
            break
    
    return edge_list()

# EVOLVE-BLOCK-END
