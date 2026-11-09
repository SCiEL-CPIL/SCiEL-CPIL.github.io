import time
import random

def construct():
    n = 15
    deadline = time.time() + 85.0
    best_edges = None
    best_count = -1

    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid(edges):
        if not edges:
            return True
        adj = build_adj(edges)
        deg = get_degrees(adj)
        # For every edge (u, v), check if there exists s~u (s!=v) and t~v (t!=u) with deg[s]==deg[t]
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Get degrees of neighbors of u (excluding v)
                degs_u = set()
                mask_u = adj[u] & ~(1 << v)
                for s in range(n):
                    if (mask_u >> s) & 1:
                        degs_u.add(deg[s])
                # Get degrees of neighbors of v (excluding u)
                degs_v = set()
                mask_v = adj[v] & ~(1 << u)
                for t in range(n):
                    if (mask_v >> t) & 1:
                        degs_v.add(deg[t])
                
                if degs_u & degs_v:
                    return False
        return True

    def try_add(u, v, adj, deg, edges, present):
        # Check if adding (u,v) keeps graph valid
        # S = neighbors of u except v
        # T = neighbors of v except u
        # Need no deg[s] == deg[t]
        
        # Optimization: Precompute sets of degrees for neighbors?
        # Since degrees change, we just compute on the fly.
        
        degs_u = set()
        mask_u = adj[u] & ~(1 << v)
        for s in range(n):
            if (mask_u >> s) & 1:
                degs_u.add(deg[s])
        
        degs_v = set()
        mask_v = adj[v] & ~(1 << u)
        for t in range(n):
            if (mask_v >> t) & 1:
                degs_v.add(deg[t])
                
        if degs_u & degs_v:
            return False
        
        # Also need to ensure s != t is handled?
        # The problem says "simple three-edge path s-u-v-t".
        # Simple path implies vertices are distinct.
        # s != u, v. t != u, v. s != t.
        # My sets contain degrees of s and t.
        # If deg[s] == deg[t], it's a violation ONLY IF s != t.
        # If s == t, then s is neighbor of u and v.
        # Path s-u-v-s is a cycle of length 3? No, s-u-v-t with s=t is s-u-v-s.
        # Is s-u-v-s a simple path? No, it's a cycle.
        # The definition of simple path requires distinct vertices.
        # So if s == t, it is NOT a simple path s-u-v-t.
        # So we should IGNORE cases where s == t.
        
        # My current check `if degs_u & degs_v` returns True if ANY degree matches.
        # This is too strict if the only matching degree is from s == t.
        
        # Let's refine:
        # Check pairs explicitly for small n?
        # Or: Check if there is a common degree d such that there exists s in S with deg[s]=d and t in T with deg[t]=d AND s != t.
        
        # Given n=15, explicit loop is fast enough and safer.
        # But we can optimize the set check:
        # If |degs_u & degs_v| > 0:
        #   For each d in intersection:
        #     Find all s in S with deg[s]==d
        #     Find all t in T with deg[t]==d
        #     If there is a pair (s,t) with s != t, invalid.
        #     If all pairs have s == t, valid (for this degree).
        
        # Actually, simpler:
        # Just iterate s in S, t in T.
        # It's O(deg(u)*deg(v)). Max 14*14 = 196.
        # This is very fast.
        
        for s in range(n):
            if s == v: continue
            if not (adj[u] >> s) & 1: continue
            for t in range(n):
                if t == u or t == s: continue
                if not (adj[v] >> t) & 1: continue
                if deg[s] == deg[t]:
                    return False
        return True

    def greedy_fill(start_edges, adj, deg, present, deadline):
        edges = list(start_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    return edges
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    if try_add(u, v, adj, deg, edges, present):
                        # Add edge
                        edges.append([u, v])
                        present.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        improved = True
        return edges

    def local_search(start_edges, deadline):
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = get_degrees(adj)
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        best_edges = [e[:] for e in edges]
        best_count = len(edges)
        
        # Try removing edges and refilling
        # Random restarts within local search
        for _ in range(50):
            if time.time() > deadline:
                break
            
            cur_edges = [e[:] for e in best_edges]
            cur_adj = build_adj(cur_edges)
            cur_deg = get_degrees(cur_adj)
            cur_present = set()
            for u, v in cur_edges:
                a, b = (u, v) if u < v else (v, u)
                cur_present.add((a, b))
                
            # Remove 1-3 random edges
            num_remove = random.randint(1, 3)
            if not cur_edges:
                continue
            for _ in range(num_remove):
                if not cur_edges:
                    break
                idx = random.randint(0, len(cur_edges) - 1)
                ru, rv = cur_edges.pop(idx)
                cur_adj[ru] &= ~(1 << rv)
                cur_adj[rv] &= ~(1 << ru)
                cur_deg[ru] -= 1
                cur_deg[rv] -= 1
                a, b = (ru, rv) if ru < rv else (rv, ru)
                cur_present.discard((a, b))
                
            # Greedy fill
            cur_edges = greedy_fill(cur_edges, cur_adj, cur_deg, cur_present, deadline)
            
            if len(cur_edges) > best_count:
                best_count = len(cur_edges)
                best_edges = [e[:] for e in cur_edges]
                
        return best_edges

    # Initial candidates
    candidates = []
    
    # 1. Empty
    candidates.append([])
    
    # 2. Random sparse
    for i in range(20):
        random.seed(i)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.1:
                    edges.append([u, v])
        candidates.append(edges)
        
    # 3. Random dense (will be heavily pruned by fix)
    for i in range(20):
        random.seed(100+i)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.4:
                    edges.append([u, v])
        candidates.append(edges)

    # Process candidates
    for cand in candidates:
        if time.time() > deadline - 5:
            break
            
        # Fix invalid: remove edges that cause violations
        # Simple greedy removal
        adj = build_adj(cand)
        deg = get_degrees(adj)
        present = set()
        for u, v in cand:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))
            
        # Iterate removing worst edges
        # Since we have fast check, we can just try to add? No, cand might be invalid.
        # We need to make it valid first.
        # Strategy: Remove edges until valid.
        edges = [e[:] for e in cand]
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            if is_valid(edges):
                break
            adj = build_adj(edges)
            deg = get_degrees(adj)
            # Find an edge that is part of a violation
            # We can just pick a random edge and remove it, then recheck?
            # Or find specific violation.
            # Let's find a specific violation edge.
            found = False
            for u in range(n):
                if found: break
                for v in range(u+1, n):
                    if found: break
                    if not (adj[u] >> v) & 1:
                        continue
                    # Check if this edge causes violation
                    # We need to check if removing it fixes a specific conflict?
                    # Easier: Just remove random edge if invalid.
                    # But that's slow if many violations.
                    # Let's find ONE violation and remove one of the involved edges.
                    # Violation: s-u-v-t with deg[s]==deg[t].
                    # Edge (u,v) is the middle.
                    # So (u,v) is a candidate for removal.
                    # Let's verify if (u,v) is indeed involved in a violation.
                    for s in range(n):
                        if s == u or s == v: continue
                        if not (adj[u] >> s) & 1: continue
                        for t in range(n):
                            if t == u or t == v or t == s: continue
                            if not (adj[v] >> t) & 1: continue
                            if deg[s] == deg[t]:
                                # Found violation involving (u,v)
                                # Remove (u,v)
                                edges = [e for e in edges if not ((e[0]==u and e[1]==v) or (e[0]==v and e[1]==u))]
                                changed = True
                                found = True
                                break
                        if found: break
                    if found: break
            
        if is_valid(edges):
            # Improve
            improved = greedy_fill(edges, build_adj(edges), get_degrees(build_adj(edges)), 
                                   set((min(u,v), max(u,v)) for u,v in edges), deadline)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved
                
            # Local search
            if time.time() < deadline - 10:
                ls_res = local_search(best_edges, deadline)
                if is_valid(ls_res) and len(ls_res) > best_count:
                    best_count = len(ls_res)
                    best_edges = ls_res

    if best_edges is None:
        return []
        
    # Final validation
    if not is_valid(best_edges):
        # Fallback to empty or single edge
        return []
        
    return [[int(u), int(v)] for u, v in best_edges]
