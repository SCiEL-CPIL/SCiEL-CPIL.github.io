"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL = list(range(1, N + 1))

def _build_conflict_graph():
    """
    Build a conflict graph: vertices 0..N-1 represent elements 1..N.
    Edge (i,j) exists if elements (i+1, j+1) cannot both be in a valid set,
    i.e., there exists k (distinct from both) such that gcd(i+1,j+1)==gcd(i+1,k+1)==gcd(j+1,k+1).
    
    Actually, two elements a,b conflict if there EXISTS some c such that {a,b,c} is bad.
    But for independent set we need: a,b can coexist iff NO c makes {a,b,c} bad.
    So edge (a,b) means: exists c != a,b with gcd(a,b)==gcd(a,c)==gcd(b,c).
    """
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    
    # conflict[i][j] = True if elements i+1 and j+1 cannot coexist
    # They cannot coexist if there exists k (1..N, k!=i+1, k!=j+1) with
    # gcd(i+1,j+1) == gcd(i+1,k) == gcd(j+1,k)
    conflict = [[False]*N for _ in range(N)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = g[a][b]
            # Check if there exists c making bad triple
            exists_bad = False
            for c in range(1, N+1):
                if c == a or c == b:
                    continue
                if g[a][c] == d and g[b][c] == d:
                    exists_bad = True
                    break
            if exists_bad:
                conflict[a-1][b-1] = True
                conflict[b-1][a-1] = True
    
    return g, conflict

def _tomita_mis(adj, n, deadline, time_budget=80):
    """
    Maximum Independent Set via branch and bound (Tomita's algorithm).
    adj: list of sets, adj[i] = set of neighbors of i (conflict graph)
    Returns the maximum independent set as a list of vertex indices.
    """
    best_set = []
    best_size = 0
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i])
    # Remap: new_index[i] = position of vertex i in the ordered list
    new_index = [0]*n
    for pos, v in enumerate(order):
        new_index[v] = pos
    
    # Build adjacency in new order
    adj_new = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            adj_new[new_index[i]].add(new_index[j])
    
    # Tomita's algorithm
    def _expand(R, P):
        """R: current independent set (list), P: candidate set (set of vertices)"""
        nonlocal best_set, best_size
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        # Pick vertex with max degree in P
        # Use a simple heuristic: pick vertex with max degree within P
        v = None
        max_deg = -1
        for u in P:
            deg = len(adj_new[u] & P)
            if deg > max_deg:
                max_deg = deg
                v = u
        
        # Branch: include v or exclude v
        # Include v
        R.append(v)
        P_new = P - {v} - adj_new[v]
        _expand(R, P_new)
        R.pop()
        
        # Exclude v
        P_excl = P - {v}
        _expand(R, P_excl)
    
    # Initial candidate set: all vertices
    P = set(range(n))
    _expand([], P)
    
    # Map back to original indices
    result = [order[i] for i in best_set]
    return result

def _greedy_mis(adj, n, deadline):
    """Greedy MIS: repeatedly pick vertex with min degree in remaining graph."""
    remaining = set(range(n))
    result = []
    adj_copy = [set(s) for s in adj]
    
    while remaining:
        if time.time() > deadline:
            break
        # Pick vertex with min degree in remaining subgraph
        best_v = None
        best_deg = n + 1
        for v in remaining:
            deg = len(adj_copy[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        result.append(best_v)
        remaining.discard(best_v)
        for u in list(remaining):
            adj_copy[u].discard(best_v)
    
    return result

def _local_search_mis(adj, n, start_set, deadline, max_rounds=200):
    """Local search on MIS: try to add vertices, or swap to improve."""
    S = set(start_set)
    best = list(S)
    best_size = len(best)
    missing = set(range(n)) - S
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a vertex
        for x in list(missing):
            # Check if x is independent from all in S
            if not (adj[x] & S):
                S.add(x)
                best = list(S)
                best_size += 1
                missing.discard(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swap
        swapped = False
        for out in list(S):
            for x in missing:
                if time.time() > deadline:
                    break
                trial = S - {out}
                if not (adj[x] & trial):
                    S = trial | {x}
                    best = list(S)
                    missing.discard(x)
                    missing.add(out)
                    swapped = True
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try 2-for-1: remove one, add two
        found = False
        for out in list(S):
            if time.time() > deadline:
                break
            trial = S - {out}
            added = []
            for x in missing:
                if x == out:
                    continue
                if not (adj[x] & (trial | set(added))):
                    trial.add(x)
                    added.append(x)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                S = trial
                best = list(S)
                best_size = len(best)
                for a in added:
                    missing.discard(a)
                missing.add(out)
                found = True
                improved = True
                break
        if improved:
            continue
        
        break
    
    return best

def _is_valid(S, g):
    S = list(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if g[a][b] == g[b][c] == g[a][c]:
                    return False
    return True

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # Build conflict graph
    g, conflict = _build_conflict_graph()
    
    if time.time() > deadline:
        return [1]
    
    # Build adjacency lists for MIS algorithms
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if conflict[i][j]:
                adj[i].add(j)
                adj[j].add(i)
    
    best_set = []
    best_size = 0
    
    # Greedy MIS with different orderings
    orderings = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda i: (i+1).bit_count()),
        sorted(range(N), key=lambda i: (0 if _is_prime(i+1) else 1, i+1)),
    ]
    
    for _ in range(5):
        rnd = list(range(N))
        random.shuffle(rnd)
        orderings.append(rnd)
    
    for order in orderings:
        if time.time() > deadline:
            break
        # Use order to bias greedy: process in order, pick min-degree among available
        S = set()
        remaining = set(range(N))
        for v in order:
            if v not in remaining:
                continue
            if not (adj[v] & S):
                S.add(v)
                remaining.discard(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Local search on best
    if time.time() < deadline:
        improved = _local_search_mis(adj, N, best_set, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Try Tomita MIS if time permits (on a reduced problem or with timeout)
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Use Tomita on the full problem with time limit
        # Tomita can be slow, so we'll use a simpler branch-and-bound with time check
        def _bb_mis(adj, n, deadline):
            best = []
            best_sz = 0
            
            # Order by degree
            degs = [len(adj[i]) for i in range(n)]
            verts = sorted(range(n), key=lambda i: degs[i])
            
            # Build sub-adjacency
            sub_adj = [set() for _ in range(n)]
            for i in range(n):
                for j in adj[i]:
                    sub_adj[i].add(j)
            
            # Use a stack-based search with time checks
            # For simplicity, do a recursive search with pruning
            def _search(R, P):
                nonlocal best, best_sz
                if time.time() > deadline:
                    return
                if not P:
                    if len(R) > best_sz:
                        best_sz = len(R)
                        best = list(R)
                    return
                if len(R) + len(P) <= best_sz:
                    return
                
                # Pick vertex
                v = next(iter(P))
                
                # Include v
                R.append(v)
                P_inc = P - {v} - sub_adj[v]
                _search(R, P_inc)
                R.pop()
                
                # Exclude v
                P_exc = P - {v}
                _search(R, P_exc)
            
            _search([], set(range(n)))
            return best
        
        bb_result = _bb_mis(adj, N, deadline)
        if len(bb_result) > best_size:
            best_size = len(bb_result)
            best_set = bb_result
    
    # Final local search
    if time.time() < deadline:
        improved = _local_search_mis(adj, N, best_set, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Convert back to 1-indexed values
    result = [v + 1 for v in best_set]
    
    # Verify
    if not _is_valid(result, g):
        # Fallback: simple greedy
        S = []
        S_set = set()
        for x in range(1, N+1):
            valid = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if gcd(x, a) == gcd(x, b) == gcd(a, b):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
        result = S
    
    return result
# EVOLVE-BLOCK-END
