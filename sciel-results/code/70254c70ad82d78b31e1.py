import time
import random

def _has_c5(adj: list[int]) -> bool:
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            x_mask = adj[u] & ~(1 << v)
            y_mask = adj[v] & ~(1 << u)
            
            x_list = []
            m = x_mask
            while m:
                lsb = m & -m
                x = lsb.bit_length() - 1
                x_list.append(x)
                m -= lsb
            
            y_list = []
            m = y_mask
            while m:
                lsb = m & -m
                y = lsb.bit_length() - 1
                y_list.append(y)
                m -= lsb
            
            for x in x_list:
                for y in y_list:
                    if x == y:
                        continue
                    common = adj[x] & adj[y]
                    forbidden = (1 << u) | (1 << v) | (1 << x) | (1 << y)
                    common &= ~forbidden
                    if common:
                        return True
    return False

def _has_independent_set_9(adj: list[int]) -> bool:
    n = len(adj)
    comp = [0] * n
    for i in range(n):
        full = (1 << n) - 1
        full_no_self = full & ~(1 << i)
        comp[i] = full_no_self & ~adj[i]
    
    found = [False]
    
    def search(clique_sz, candidates):
        if found[0]:
            return
        if clique_sz >= 9:
            found[0] = True
            return
        if not candidates:
            return
        if clique_sz + candidates.bit_count() < 9:
            return
        
        c_list = []
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            c_list.append(v)
            m -= lsb
        
        if not c_list:
            return
            
        c_list.sort(key=lambda v: (comp[v] & candidates).bit_count(), reverse=True)
        v = c_list[0]
        
        new_candidates = candidates & comp[v]
        search(clique_sz + 1, new_candidates)
        
        if found[0]:
            return
        
        new_candidates2 = candidates & ~(1 << v)
        search(clique_sz, new_candidates2)
        
    search(0, (1 << n) - 1)
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_fallback(n):
        edges = []
        for b in range(n // 4):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    fallback = get_fallback(32)
    
    for target_n in [34, 33]:
        if time.time() > deadline:
            break
        
        found_valid = False
        while time.time() < deadline and not found_valid:
            # Base: 8 K4s
            adj = [0] * target_n
            for b in range(8):
                base = b * 4
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj[base + i] |= (1 << (base + j))
                        adj[base + j] |= (1 << (base + i))
            
            # Add random edges for new vertices (32, 33)
            for v in range(32, target_n):
                # Connect to random subset of existing vertices
                # Try to connect to 1 vertex per block?
                for b in range(8):
                    base = b * 4
                    # Pick 1-2 random vertices in block
                    k = random.randint(1, 2)
                    candidates = [base + i for i in range(4)]
                    random.shuffle(candidates)
                    for u in candidates[:k]:
                        if random.random() < 0.7: # Probability of adding edge
                            adj[v] |= (1 << u)
                            adj[u] |= (1 << v)
            
            # Check
            if not _has_c5(adj) and not _has_independent_set_9(adj):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
                found_valid = True
        
    return fallback
