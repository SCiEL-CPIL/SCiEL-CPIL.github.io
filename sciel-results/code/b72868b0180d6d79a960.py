import time
import random

def has_k4_count(adj, n):
    """
    Counts the number of K4s in the graph.
    """
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    if w > v:
                        # Check for x in N(u) & N(v) & N(w) with x > w
                        rest = common & adj[w]
                        r = rest
                        while r:
                            lsb_r = r & -r
                            x = lsb_r.bit_length() - 1
                            if x > w:
                                count += 1
                            r ^= lsb_r
                    c ^= lsb
    return count

def find_independent_set_6(adj, n):
    """
    Finds an independent set of size 6 if it exists.
    Returns list of vertices or None.
    """
    comp = [0] * n
    for i in range(n):
        mask = (1 << n) - 1
        mask ^= (1 << i)
        comp[i] = mask & ~adj[i]
        
    def dfs(candidates_mask, size, current_set):
        if size == 0:
            return current_set
        if candidates_mask == 0:
            return None
        if size == 1:
            lsb = candidates_mask & -candidates_mask
            v = lsb.bit_length() - 1
            return current_set + [v]
        
        # Heuristic: Pick lowest index for determinism
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Include v
        new_cands = candidates_mask & comp[v]
        res = dfs(new_cands, size - 1, current_set + [v])
        if res:
            return res
        
        # Exclude v
        candidates_mask ^= lsb
        res = dfs(candidates_mask, size, current_set)
        if res:
            return res
        
        return None

    full_mask = (1 << n) - 1
    return dfs(full_mask, 6, [])

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def get_circulant(n, distances):
        adj = [0] * n
        for d in distances:
            for i in range(n):
                j = (i + d) % n
                if i != j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for i in range(n):
            adj[i] &= ~(1 << i)
        return adj

    # Deterministic candidates for n=35 and n=36
    candidates = {
        36: [
            [1, 2, 3, 5, 7, 10, 15],
            [1, 2, 3, 4, 5, 7, 10, 15],
            [1, 2, 3, 4, 5, 6, 7, 10, 15],
            [1, 2, 4, 5, 7, 10, 15],
        ],
        35: [
            [1, 2, 3, 5, 7, 10, 15],
            [1, 2, 3, 4, 5, 7, 10, 15],
            [1, 2, 3, 4, 5, 6, 7, 10, 15],
            [1, 2, 4, 5, 7, 10, 15],
            [1, 2, 3, 5, 6, 8, 10, 15],
        ]
    }
    
    for n in [36, 35]:
        if time.time() - start_time > time_limit:
            break
            
        for dists in candidates[n]:
            if time.time() - start_time > time_limit:
                break
            
            adj = get_circulant(n, dists)
            
            # Check initial
            k4c = has_k4_count(adj, n)
            indep = find_independent_set_6(adj, n)
            
            if k4c == 0 and indep is None:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # Greedy Repair
            current_adj = adj[:]
            
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                k4c = has_k4_count(current_adj, n)
                indep = find_independent_set_6(current_adj, n)
                
                if k4c == 0 and indep is None:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if current_adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Generate moves
                moves = []
                
                if k4c > 0:
                    # Find one K4 to break
                    k4_found = None
                    for u in range(n):
                        if k4_found: break
                        for v in range(u + 1, n):
                            if k4_found: break
                            if current_adj[u] & (1 << v):
                                common = current_adj[u] & current_adj[v]
                                c = common
                                while c:
                                    if k4_found: break
                                    lsb = c & -c
                                    w = lsb.bit_length() - 1
                                    if w > v:
                                        rest = common & current_adj[w]
                                        r = rest
                                        while r:
                                            lsb_r = r & -r
                                            x = lsb_r.bit_length() - 1
                                            if x > w:
                                                k4_found = (u, v, w, x)
                                                break
                                            r ^= lsb_r
                                    c ^= lsb
                    
                    if k4_found:
                        u, v, w, x = k4_found
                        pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                        for a, b in pairs:
                            if current_adj[a] & (1 << b):
                                moves.append((a, b, -1))
                
                if indep:
                    for i in range(len(indep)):
                        for j in range(i+1, len(indep)):
                            a = indep[i]
                            b = indep[j]
                            if not (current_adj[a] & (1 << b)):
                                moves.append((a, b, 1))
                                
                if not moves:
                    # Stuck. Random flip
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a != b:
                        if current_adj[a] & (1 << b):
                            moves.append((a, b, -1))
                        else:
                            moves.append((a, b, 1))
                    else:
                        break
                
                # Pick a move
                a, b, action = random.choice(moves)
                
                if action == -1:
                    current_adj[a] ^= (1 << b)
                    current_adj[b] ^= (1 << a)
                else:
                    current_adj[a] |= (1 << b)
                    current_adj[b] |= (1 << a)

    return {'n': 1, 'edges': []}
