"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def _check_validity(adj, deg):
    """Check validity given adjacency lists and degrees. Returns True if valid."""
    n = len(adj)
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
                    if deg[s] == deg[t]:
                        return False
    return True

def _build_adj_deg(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    return adj, deg

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    
    # Base: K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    base_edges = []
    for a in part_a:
        for b in part_b:
            base_edges.append((a, b))
    
    base_adj, base_deg = _build_adj_deg(base_edges, n)
    assert _check_validity(base_adj, base_deg)
    
    best_edges = base_edges[:]
    best_count = len(best_edges)
    
    # Candidate edges to add: within A, within B
    candidates = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i + 1, 15):
            candidates.append((i, j))
    
    # Greedy with multiple random restarts
    random.seed(12345)
    
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        
        cur_edges = best_edges[:]
        cur_adj, cur_deg = _build_adj_deg(cur_edges, n)
        
        cands = candidates[:]
        random.shuffle(cands)
        
        for ce in cands:
            if time.time() > deadline - 10:
                break
            u, v = ce
            # Add edge
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            cur_deg[u] += 1
            cur_deg[v] += 1
            cur_edges.append(ce)
            
            if _check_validity(cur_adj, cur_deg):
                best_edges = cur_edges[:]
                best_count = len(best_edges)
            else:
                # Remove
                cur_adj[u].discard(v)
                cur_adj[v].discard(u)
                cur_deg[u] -= 1
                cur_deg[v] -= 1
                cur_edges.pop()
    
    # Try different bipartitions too
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        # Random bipartition
        parts = list(range(n))
        random.shuffle(parts)
        k = random.randint(5, 10)
        pa = parts[:k]
        pb = parts[k:]
        
        te = []
        for a in pa:
            for b in pb:
                te.append((a, b))
        
        ta, td = _build_adj_deg(te, n)
        if _check_validity(ta, td):
            if len(te) > best_count:
                best_edges = te[:]
                best_count = len(te)
            
            # Try to add intra-part edges
            cands2 = []
            for i in range(len(pa)):
                for j in range(i+1, len(pa)):
                    cands2.append((pa[i], pa[j]))
            for i in range(len(pb)):
                for j in range(i+1, len(pb)):
                    cands2.append((pb[i], pb[j]))
            
            random.shuffle(cands2)
            for ce in cands2:
                if time.time() > deadline - 10:
                    break
                u, v = ce
                ta[u].add(v)
                ta[v].add(u)
                td[u] += 1
                td[v] += 1
                te.append(ce)
                if _check_validity(ta, td):
                    if len(te) > best_count:
                        best_edges = te[:]
                        best_count = len(te)
                else:
                    ta[u].discard(v)
                    ta[v].discard(u)
                    td[u] -= 1
                    td[v] -= 1
                    te.pop()
    
    # Final result
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
