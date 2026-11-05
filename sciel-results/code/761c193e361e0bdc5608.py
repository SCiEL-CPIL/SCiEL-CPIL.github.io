import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Seed graph: Andrasfai graph And(9)
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_n = n_seed
    best_edges = edges_seed
    
    def check_validity(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Triangle free check
        for u in range(n):
            for v in adj[u]:
                if v < u: continue
                if adj[u].intersection(adj[v]):
                    return False
                    
        # Independent set check (size 10)
        masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u != v and v not in adj[u]:
                    mask |= (1 << v)
            masks[u] = mask
            
        def dfs(candidates, count):
            if count == 10:
                return True
            if candidates == 0:
                return False
            if count + bin(candidates).count('1') < 10:
                return False
                
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = bin(candidates & masks[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
                
            if best_v == -1:
                return False
                
            if dfs(candidates & masks[best_v], count + 1):
                return True
            if dfs(candidates & ~(1 << best_v), count):
                return True
            return False
            
        if dfs((1 << n) - 1, 0):
            return False
        return True

    def try_find_graph(n):
        for _ in range(20): # 20 restarts
            if time.time() > deadline - 5:
                return None
            
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            
            adj = [set() for _ in range(n)]
            edges = []
            
            for u, v in all_edges:
                if not adj[u].intersection(adj[v]):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    
            # Local search
            # Find safe non-edges
            safe_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if not adj[u].intersection(adj[v]):
                            safe_edges.append((u, v))
                            
            # Add as many safe edges as possible
            while safe_edges:
                if time.time() > deadline - 5:
                    return None
                idx = random.randint(0, len(safe_edges) - 1)
                u, v = safe_edges.pop(idx)
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Invalidate safe edges that become unsafe
                # An edge (a,b) is unsafe if a in adj[b] (new) or b in adj[a] (new)
                # Specifically, if we added (u,v), then for any w in adj[u], (w,v) is now unsafe?
                # No, (w,v) is unsafe if w in adj[v].
                # Before adding (u,v), w was not in adj[v] (otherwise (u,v) wouldn't be safe? No, (u,v) safe means adj[u] & adj[v] is empty. So w in adj[u] implies w not in adj[v]).
                # After adding (u,v), v is in adj[u] and u is in adj[v].
                # So any edge (w, v) where w in adj[u] is now unsafe because w in adj[u] and v in adj[u]... wait.
                # Unsafe condition for (a,b) is adj[a] & adj[b] != empty.
                # If we add (u,v), adj[u] gains v, adj[v] gains u.
                # Consider a safe edge (a,b) in the list.
                # If a == u or b == u, and the other vertex is v? No, (u,v) was removed from list.
                # If a == u, and b != v. Is (u,b) unsafe now?
                # adj[u] now contains v. adj[b] contains ...
                # If b in adj[v] (old), then v in adj[b].
                # Then adj[u] (new) contains v, and adj[b] contains v. So v is common neighbor.
                # So (u,b) is unsafe if b in adj[v] (old).
                # Similarly (v,a) is unsafe if a in adj[u] (old).
                
                # To be safe and simple, we can just re-calculate safe_edges if we add many edges.
                # Or we can just check validity after adding all.
                # Since we are adding edges, alpha can only decrease.
                # We just need to ensure we don't create triangles.
                # The "safe" condition guarantees no triangles.
                # So we can just add all safe edges we found initially?
                # No, adding one might make another unsafe.
                # But if we only add edges that are safe *at the time of addition*, we are fine.
                # However, our list `safe_edges` is static.
                # So we must filter it.
                
                # Optimization: Instead of complex filtering, just restart local search if list gets large?
                # Or just do a full re-scan of non-edges every 10 additions.
                
                if len(edges) % 10 == 0:
                    # Re-scan for new safe edges
                    new_safe = []
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if v2 not in adj[u2]:
                                if not adj[u2].intersection(adj[v2]):
                                    new_safe.append((u2, v2))
                    safe_edges = new_safe

            # Final check
            if check_validity(n, edges):
                return edges
                
        return None

    for n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break
        edges = try_find_graph(n)
        if edges:
            best_n = n
            best_edges = edges
            if n == 40:
                break

    return {'n': best_n, 'edges': best_edges}
