import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 16
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_neighbors = [[] for _ in range(8)]
    for u, v in q3_edges:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)

    def has_q3(adj):
        """
        adj: list of 16 ints, bit j set in adj[i] means edge (i,j) exists.
        Returns True if Q3 subgraph exists.
        """
        for g0 in range(n):
            assign = {0: g0}

            def backtrack(qv):
                if qv == 8:
                    return True

                used = set(assign.values())

                for gv in range(n):
                    if gv in used:
                        continue

                    ok = True
                    for nq in q3_neighbors[qv]:
                        if nq in assign:
                            gn = assign[nq]
                            if not (adj[gv] >> gn) & 1:
                                ok = False
                                break
                    if not ok:
                        continue

                    assign[qv] = gv
                    if backtrack(qv + 1):
                        return True
                    del assign[qv]

                return False

            if backtrack(1):
                return True

        return False

    def find_q3(adj):
        """
        Returns a list of 8 vertices forming a Q3, or None.
        """
        for g0 in range(n):
            assign = {0: g0}
            
            def backtrack(qv):
                if qv == 8:
                    return list(assign.values())
                
                used = set(assign.values())
                
                for gv in range(n):
                    if gv in used:
                        continue
                    
                    ok = True
                    for nq in q3_neighbors[qv]:
                        if nq in assign:
                            gn = assign[nq]
                            if not (adj[gv] >> gn) & 1:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    assign[qv] = gv
                    res = backtrack(qv + 1)
                    if res:
                        return res
                    del assign[qv]
                
                return None

            res = backtrack(1)
            if res:
                return res

        return None

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges_count = 0
    best_adj = None

    # Strategy: Start with K8 U K8 and add cross edges
    # A = 0..7, B = 8..15
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(trial)
        
        adj = [0] * n
        
        # Add K8 on 0..7
        for i in range(8):
            for j in range(i+1, 8):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Add K8 on 8..15
        for i in range(8, 16):
            for j in range(i+1, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Add random cross edges
        p = 0.5 + random.random() * 0.3 # 0.5 to 0.8
        for i in range(8):
            for j in range(8, 16):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        
        # Repair: Remove edges to make Q3-free
        # We prefer removing cross edges
        cross_edges = [(i, j) for i in range(8) for j in range(8, 16) if (adj[i] >> j) & 1]
        internal_edges_A = [(i, j) for i in range(8) for j in range(i+1, 8)]
        internal_edges_B = [(i, j) for i in range(8, 16) for j in range(i+1, 16)]
        
        # Shuffle for random removal order
        random.shuffle(cross_edges)
        random.shuffle(internal_edges_A)
        random.shuffle(internal_edges_B)
        
        # Priority: Remove cross edges first, then internal
        # But removing internal edges is more effective?
        # Let's just remove any edge from the found Q3.
        
        steps = 0
        while time.time() < deadline - 5 and steps < 1000:
            steps += 1
            q3 = find_q3(adj)
            if not q3:
                break
            
            # Pick a random edge from the Q3
            # The Q3 vertices are q3[0]..q3[7]
            # The edges of Q3 in this labeling are q3_edges
            # We need to map these to graph edges
            # q3_edges are (u,v) in Q3 labeling.
            # The edge in G is (q3[u], q3[v])
            
            edges_in_q3 = []
            for u, v in q3_edges:
                gu, gv = q3[u], q3[v]
                if gu < gv:
                    edges_in_q3.append((gu, gv))
                else:
                    edges_in_q3.append((gv, gu))
            
            # Randomly pick one edge to remove
            u, v = random.choice(edges_in_q3)
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            count -= 1
            
        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]
            
        # Local search: Try to add back edges
        if best_adj is not None and time.time() < deadline - 2:
            adj_ls = best_adj[:]
            count_ls = best_edges_count
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for u, v in all_edges:
                    if (adj_ls[u] >> v) & 1:
                        continue
                    adj_ls[u] |= (1 << v)
                    adj_ls[v] |= (1 << u)
                    if not has_q3(adj_ls):
                        count_ls += 1
                        improved = True
                    else:
                        adj_ls[u] ^= (1 << v)
                        adj_ls[v] ^= (1 << u)
                        
            if count_ls > best_edges_count:
                best_edges_count = count_ls
                best_adj = adj_ls

    if best_adj is None:
        return []
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

construct()
