import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 110.0
        
        def get_edges(adj, n):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return edges

        def find_k4(n, adj):
            # Returns (u, v, w, x) or None
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        m = common
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            m ^= lsb
                            if (common >> w) & 1:
                                # w is in common, check if w and another common node are connected
                                # Actually, we need x in common such that w-x is edge.
                                common_w = common & adj[w]
                                common_w &= ~(1 << w)
                                if common_w:
                                    lsb2 = common_w & -common_w
                                    x = lsb2.bit_length() - 1
                                    return (u, v, w, x)
            return None

        def find_i6(n, adj):
            # Returns list of 6 vertices or None
            # Use a simple backtracking or random sampling?
            # For n=25, we can iterate combinations? 177k is okay.
            # But we can do better:
            # Randomly sample 6 vertices. If they form an I6, return them.
            # If we don't find one in 1000 tries, assume none? Risky.
            # Let's do a deterministic check for small n.
            
            # Heuristic: 
            # If n is small, check all.
            # For n=25, 177k checks. Each check:
            # Take 6 vertices v0..v5.
            # Check all pairs are non-edges.
            # This is 15 checks per set. 177k * 15 = 2.6M ops.
            # In Python, 2.6M ops is ~0.1s.
            # So doing this every 100 steps is fine.
            
            # We'll use a randomized check first for speed in the loop, 
            # and exact check only when K4 count is 0.
            
            # For the repair step, we just need *one* I6.
            # Random sampling is good.
            for _ in range(1000):
                verts = random.sample(range(n), 6)
                is_indep = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[verts[i]] >> verts[j]) & 1:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return verts
            return None

        def exact_check(n, adj):
            # Final verification
            if find_k4(n, adj) is not None:
                return False
            # Exact I6 check
            from itertools import combinations
            for combo in combinations(range(n), 6):
                is_indep = True
                for i in range(6):
                    for j in range(i+1, 6):
                        if (adj[combo[i]] >> combo[j]) & 1:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return False
            return True

        best_n = 0
        best_graph = None

        # Try n=25, 30, 35
        for target_n in [25, 30, 35]:
            if time.time() > deadline - 20:
                break
            
            found = False
            for restart in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Init
                adj = [0] * target_n
                p = 0.3
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < p:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Repair Loop
                for step in range(5000):
                    if time.time() > deadline - 5:
                        break
                    
                    k4 = find_k4(target_n, adj)
                    if k4:
                        # Remove an edge from K4
                        u, v, w, x = k4
                        # Pick a random edge from the K4 to remove
                        # Edges: uv, uw, ux, vw, vx, wx
                        edge_pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                        u1, v1 = random.choice(edge_pairs)
                        adj[u1] ^= (1 << v1)
                        adj[v1] ^= (1 << u1)
                    else:
                        # No K4, check I6
                        i6 = find_i6(target_n, adj)
                        if i6:
                            # Add an edge to break I6
                            u1, v1 = random.sample(i6, 2)
                            # Ensure not already connected (they aren't, since it's I6)
                            adj[u1] |= (1 << v1)
                            adj[v1] |= (1 << u1)
                        else:
                            # No K4, no I6 found (probabilistically)
                            # Do exact check
                            if exact_check(target_n, adj):
                                best_n = target_n
                                best_graph = adj[:]
                                found = True
                                break
                            else:
                                # Exact check failed, meaning we missed an I6 in random sampling
                                # Force a random flip to escape
                                u1 = random.randint(0, target_n-1)
                                v1 = random.randint(0, target_n-1)
                                if u1 != v1:
                                    adj[u1] ^= (1 << v1)
                                    adj[v1] ^= (1 << u1)
                
                if found:
                    break
            
            if found:
                break

        if best_graph is None:
            # Fallback: n=5 C5
            adj = [0] * 5
            for i in range(5):
                adj[i] |= (1 << ((i+1)%5))
                adj[i] |= (1 << ((i-1)%5))
            return {'n': 5, 'edges': get_edges(adj, 5)}
        else:
            return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
