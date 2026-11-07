import time
import random

N = 15
DEADLINE = 55.0  # Safety margin under the 90s limit

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    # For every middle edge (u, v) with u < v, for every simple 3-edge path
    # s - u - v - t  (s in adj[u]\{v}, t in adj[v]\{u}, s != t),
    # require deg[s] != deg[t].
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _build_adj(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)
    
    # Initial best: K_{7,8}
    # Partition A: 0..6, B: 7..14
    best_edges = [(a, b) for a in range(7) for b in range(7, 15)]
    best_len = len(best_edges)
    
    # Helper to update best
    def consider(cand_edges):
        nonlocal best_edges, best_len
        if len(cand_edges) > best_len:
            best_len = len(cand_edges)
            best_edges = cand_edges

    # Phase 1: Local search starting from K_{7,8}
    # We try to add edges. If invalid, we try to fix by removing edges.
    # We also try random restarts from other seeds.
    
    seeds = [
        best_edges, # K_{7,8}
        [(a, b) for a in range(8) for b in range(8, 15)], # K_{8,7}
        [(0, i) for i in range(1, N)], # Star
        [] # Empty
    ]
    
    # Add some random dense seeds
    for _ in range(5):
        rnd_edges = []
        for e in all_edges:
            if rng.random() < 0.5:
                rnd_edges.append(e)
        seeds.append(rnd_edges)

    for seed in seeds:
        if time.time() - t0 > DEADLINE:
            break
        
        # Greedy repair/add
        current_edges = list(seed)
        adj = _build_adj(current_edges)
        deg = _degrees(adj)
        
        # First, make it valid by removing conflicting edges
        # This is a simple repair loop
        valid = False
        while not valid:
            valid = _is_valid(adj, deg)
            if valid:
                break
            # Find a violation
            viol_edge = None
            for u in range(N):
                au = adj[u]
                for v in au:
                    if v <= u:
                        continue
                    av = adj[v]
                    for s in au:
                        if s == v:
                            continue
                        ds = deg[s]
                        for t in av:
                            if t == u or t == s:
                                continue
                            if ds == deg[t]:
                                # Violation found on path s-u-v-t
                                # Remove a random edge from the path to break it or change degrees
                                # Options: (s,u), (u,v), (v,t)
                                opts = [(s,u), (u,v), (v,t)]
                                e_rem = rng.choice(opts)
                                a, b = e_rem
                                adj[a].remove(b)
                                adj[b].remove(a)
                                deg[a] -= 1
                                deg[b] -= 1
                                # Remove from current_edges
                                if (a,b) in current_edges:
                                    current_edges.remove((a,b))
                                elif (b,a) in current_edges:
                                    current_edges.remove((b,a))
                                viol_edge = (a,b)
                                break
                        if viol_edge: break
                    if viol_edge: break
                if viol_edge: break
            if viol_edge is None:
                break # Should not happen if valid is False
            
        # Now current_edges is valid. Try to add more.
        consider(current_edges)
        
        # Greedy add
        remaining = [e for e in all_edges if e not in current_edges]
        rng.shuffle(remaining)
        for e in remaining:
            if time.time() - t0 > DEADLINE:
                break
            u, v = e
            # Check if adding (u,v) keeps validity
            # Fast check:
            # New deg: deg[u]+1, deg[v]+1
            # New paths involving (u,v):
            # 1. s-u-v-t: s in N(u)\{v}, t in N(v)\{u}. deg[s] != deg[t].
            #    Note: deg[s] and deg[t] are old degrees unless s==u or s==v (impossible for simple path s-u-v-t with s!=v, t!=u).
            #    Wait, if s is neighbor of u, s!=v. t is neighbor of v, t!=u.
            #    The degrees of s and t do NOT change when we add (u,v) unless s==u or s==v?
            #    No, s is a neighbor of u. If we add (u,v), deg(u) and deg(v) change.
            #    deg(s) changes only if s is u or v. But s!=v and s!=u (since s-u edge exists and s!=u? No, s is neighbor of u, so s!=u is true for simple graph? No, s can be u? No, s is in N(u), so s!=u).
            #    So deg(s) and deg(t) remain the same as in the current graph.
            #    So we just need to check if for all s in N(u)\{v} and t in N(v)\{u}, s!=t, deg[s] != deg[t].
            #    This is a subset of the full check.
            
            # Also, adding (u,v) creates new paths where (u,v) is NOT the middle edge?
            # No, the condition is "for every simple 3-edge path s-u-v-t".
            # The middle edge is (u,v) in the notation s-u-v-t?
            # The task says: "path s-u-v-t on edge (u, v)".
            # This implies (u,v) is the middle edge.
            # So we only need to check paths where (u,v) is the middle edge.
            # What about paths where (u,v) is an end edge?
            # e.g. u-v-w-x. Middle edge is (v,w).
            # The check for middle edge (v,w) will cover this.
            # So when adding (u,v), we only need to ensure that no NEW violation is created where (u,v) is the middle edge.
            # Existing paths where (u,v) is an end edge were already valid (or invalid, but we assume current is valid).
            # Do degrees of endpoints of existing paths change?
            # Yes, deg(u) and deg(v) change.
            # This might break existing valid paths where u or v is an endpoint.
            # Example: Path s-x-y-u. Middle edge (x,y). Endpoints s, u.
            # Before: deg[s] != deg[u].
            # After: deg[u] increases by 1.
            # If deg[s] == deg[u] + 1, then after addition deg[s] == deg[u]_new. Violation!
            
            # So we must re-check the full graph or be very careful.
            # Given N=15, full check is fast.
            
            trial_edges = current_edges + [e]
            if _valid(trial_edges):
                current_edges = trial_edges
                consider(current_edges)

    # Phase 2: More aggressive local search
    # Try swapping edges: remove one, add one.
    # Or remove one, add two.
    
    for _ in range(100):
        if time.time() - t0 > DEADLINE:
            break
        
        base = best_edges[:]
        # Remove a random edge
        if base:
            idx = rng.randint(0, len(base)-1)
            rem_edge = base[idx]
            base.pop(idx)
            
            # Try to add random edges greedily
            remaining = [e for e in all_edges if e not in base]
            rng.shuffle(remaining)
            for e in remaining:
                if time.time() - t0 > DEADLINE:
                    break
                trial = base + [e]
                if _valid(trial):
                    base = trial
            
            consider(base)
            
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result
