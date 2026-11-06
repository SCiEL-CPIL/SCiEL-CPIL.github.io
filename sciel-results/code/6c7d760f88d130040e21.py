import time
import random
from itertools import combinations

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    Strategy:
    1. Hardcoded valid 32-vertex graph (Union of 8 K4s). Score = 32/28 = 1.1428...
    2. Attempt to find a valid 33, 34, or 35-vertex graph via "Grow and Fix" search.
       - If found and verified, return it.
       - Otherwise, return the hardcoded 32-vertex graph.
    Time limit guard ensures we finish well before 120s.
    """
    
    # 1. Base solution: 32 vertices (8 disjoint K4s)
    def get_base_graph():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    base_graph = get_base_graph()
    
    # Verification and Search Helpers
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        adj_mask = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj, adj_mask

    def has_c5(adj, n):
        # Returns True if C5 exists
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    for x in adj[w]:
                        if x in (u, v): continue
                        for y in adj[x]:
                            if y in (u, v, w): continue
                            if y in adj[u]:
                                return True
        return False

    def find_c5(adj, n):
        # Returns one C5 as a list of 5 vertices if found, else None
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    for x in adj[w]:
                        if x in (u, v): continue
                        for y in adj[x]:
                            if y in (u, v, w): continue
                            if y in adj[u]:
                                return [u, v, w, x, y]
        return None

    def has_is_k(adj_mask, k, n):
        # Returns True if independent set of size k exists
        def dfs(cand, rem):
            if rem == 0: return True
            if cand == 0: return False
            if cand.bit_count() < rem: return False
            
            # Pick vertex with max degree in cand
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1: return False
            
            # Include v
            new_cand_incl = cand & ~adj_mask[v] & ~(1 << v)
            if dfs(new_cand_incl, rem - 1):
                return True
            
            # Exclude v
            new_cand_excl = cand & ~(1 << v)
            if dfs(new_cand_excl, rem):
                return True
            
            return False

        all_mask = (1 << n) - 1
        return dfs(all_mask, k)

    def find_is_k(adj_mask, k, n):
        # Returns an independent set of size k if found, else None
        result = []
        def dfs(cand, rem, current):
            if rem == 0:
                result[:] = current
                return True
            if cand == 0: return False
            if cand.bit_count() < rem: return False
            
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1: return False
            
            # Include v
            new_cand_incl = cand & ~adj_mask[v] & ~(1 << v)
            current.append(v)
            if dfs(new_cand_incl, rem - 1, current):
                return True
            current.pop()
            
            # Exclude v
            new_cand_excl = cand & ~(1 << v)
            if dfs(new_cand_excl, rem, current):
                return True
            
            return False

        all_mask = (1 << n) - 1
        if dfs(all_mask, k, result):
            return result
        return None

    def has_path_len_4(adj, u, v):
        # Check if there is a path of length 4 from u to v
        # BFS from u
        if u == v: return False
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= 4: continue
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = dist + 1
                    if nb == v and dist + 1 == 4:
                        return True
                    queue.append((nb, dist + 1))
        return False

    def try_add_edge_safe(adj, adj_mask, u, v, n):
        # Returns True if adding edge (u,v) does not create a C5
        if u in adj[v]: return True # Already connected
        # Check if u and v are at distance 4
        if has_path_len_4(adj, u, v):
            return False
        return True

    # Search Logic
    deadline = time.time() + 110.0
    random.seed(42)
    
    best_n = 32
    best_edges = base_graph['edges']
    
    # Try to find n=35, 34, 33
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and time.time() < deadline - 5 and not found:
            attempts += 1
            
            # Initialize graph
            n = target_n
            adj = [set() for _ in range(n)]
            adj_mask = [0] * n
            
            # Build up
            valid = True
            for v in range(n):
                # Connect v to random previous vertices
                # Bias towards higher degree to reduce alpha?
                # Or random?
                if v == 0:
                    continue
                
                # Choose subset of previous vertices to connect to
                # Random probability 0.5
                conns = []
                for u in range(v):
                    if random.random() < 0.4:
                        conns.append(u)
                
                # Add edges
                for u in conns:
                    adj[v].add(u)
                    adj[u].add(v)
                    adj_mask[v] |= (1 << u)
                    adj_mask[u] |= (1 << v)
                
                # Fix C5s
                # This might be slow if done naively.
                # We'll just check and remove.
                # To speed up, we only check C5s involving v?
                # A C5 involving v must use v.
                # So we can check paths of length 4 from v to its neighbors?
                # Or just run global check?
                # Global check is O(n*d^4). For n=35, d~10, 35*10000 = 350k.
                # Doing this for each vertex (35 times) is 12M ops. Feasible.
                
                while True:
                    c5 = find_c5(adj, n)
                    if c5 is None:
                        break
                    
                    # Remove a random edge from the C5
                    # Do not remove edges incident to v if possible?
                    # Just pick random edge from c5
                    edge_idx = random.randint(0, 4)
                    u = c5[edge_idx]
                    w = c5[(edge_idx + 1) % 5]
                    
                    # Remove edge
                    adj[u].discard(w)
                    adj[w].discard(u)
                    adj_mask[u] &= ~(1 << w)
                    adj_mask[w] &= ~(1 << u)
                
                # Check Alpha
                if has_is_k(adj_mask, 9, n):
                    # Try to fix alpha by adding edges
                    fixed_alpha = False
                    is_set = find_is_k(adj_mask, 9, n)
                    if is_set:
                        # Try to add edges within is_set
                        added_any = False
                        for i in range(len(is_set)):
                            for j in range(i+1, len(is_set)):
                                u = is_set[i]
                                v2 = is_set[j]
                                if try_add_edge_safe(adj, adj_mask, u, v2, n):
                                    # Add edge
                                    adj[u].add(v2)
                                    adj[v2].add(u)
                                    adj_mask[u] |= (1 << v2)
                                    adj_mask[v2] |= (1 << u)
                                    added_any = True
                                    # Re-check alpha?
                                    # Adding an edge reduces alpha or keeps it same.
                                    # It might create a C5? We checked safe.
                                    # But it might create a NEW independent set of size 9?
                                    # No, adding edges only reduces the size of any independent set.
                                    # So if we break this IS, we might be safe.
                                    # But we should verify.
                                    pass
                        
                        if added_any:
                            # Re-verify C5 and Alpha
                            if not has_c5(adj, n) and not has_is_k(adj_mask, 9, n):
                                fixed_alpha = True
                            else:
                                # Failed to fix cleanly, or introduced C5?
                                # try_add_edge_safe should prevent C5.
                                # But maybe alpha is still 9?
                                # We added edges, so alpha should be smaller.
                                # If has_is_k is still True, we need more edges.
                                # For now, mark as failed if not fixed immediately.
                                # We can loop, but let's keep it simple.
                                pass
                    
                    if not fixed_alpha:
                        valid = False
                        break
            
            if valid:
                # Final verification
                if not has_c5(adj, n) and not has_is_k(adj_mask, 9, n):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break
    
    if best_n > 32:
        return {'n': best_n, 'edges': best_edges}
    else:
        return base_graph
