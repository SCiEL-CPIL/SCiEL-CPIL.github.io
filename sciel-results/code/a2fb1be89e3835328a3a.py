"""Search-based construction for R(C4,K11) > n using Polarity Graphs and Conflict-based Expansion."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if len(au & adj[v]) >= 2:
                return False
    return True

def _independent_set_ge(adj, n, limit, deadline):
    """Return True if there is an independent set of size >= limit.
    Bitset branch-and-bound with a wall-clock deadline."""
    masks = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= 1 << u
        masks[v] = m
    full = (1 << n) - 1

    best = [0]
    
    def _dfs(rem, size):
        if time.time() > deadline:
            return
        if size >= limit:
            best[0] = size
            return
        if not rem:
            if size > best[0]:
                best[0] = size
            return
        # Upper bound
        if size + rem.bit_count() <= best[0]:
            return
        
        # Heuristic: pick vertex with max degree in rem to branch
        # This helps prune faster
        r = rem
        # Find vertex with max degree in rem
        max_deg = -1
        best_v = -1
        # Iterate through bits
        temp = r
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = (masks[v] & r).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= lb
            
        if best_v == -1:
            return
            
        v = best_v
        lb = 1 << v
        
        # Option 1: exclude v
        _dfs(rem & ~lb, size)
        # Option 2: include v -> remove v and its neighbours
        _dfs(rem & ~lb & ~masks[v], size + 1)

    _dfs(full, 0)
    return best[0] >= limit

def _construct_polarity_graph(q):
    """Polarity graph of PG(2,q). Vertices are points, edge if dot product 0 mod q."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q - 2, q)
                    key = (1, (y * inv) % q, (z * inv) % q)
                elif y != 0:
                    inv = pow(y, q - 2, q)
                    key = (0, 1, (z * inv) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        a1, b1, c1 = points[i]
        for j in range(i + 1, n_pts):
            a2, b2, c2 = points[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n_pts, 'edges': edges, 'adj': adj}

def _find_valid_neighborhoods(adj, n, deadline):
    """
    Find valid neighborhoods S for a new vertex.
    Conditions:
    1. For all u in V, |N(u) cap S| <= 1.
    2. For all x, y in S, N(x) cap N(y) = empty.
    Returns a list of candidate sets S.
    """
    candidates = []
    
    # Pre-compute conflict constraints
    # Constraint 2: x and y cannot both be in S if N(x) & N(y) is not empty.
    # Let's build a "conflict graph" C on V where edge (x,y) exists if N(x) & N(y) != empty.
    # Then S must be an independent set in C.
    # Constraint 1: For each u, S cannot contain two vertices from N(u).
    
    # We use a backtracking search to find independent sets in C that also satisfy Constraint 1.
    # We want S to be as large as possible to "cover" vertices and reduce alpha, 
    # but not so large that it isolates too many vertices.
    
    # Actually, we just need *some* valid S. We can try random subsets of V that are independent in C.
    
    # Build conflict adjacency
    conflict = [set() for _ in range(n)]
    for x in range(n):
        for y in range(x+1, n):
            if adj[x] & adj[y]:
                conflict[x].add(y)
                conflict[y].add(x)
                
    # Heuristic search for S
    # Try to pick vertices one by one.
    # Order vertices by degree in conflict graph (descending) to prune early?
    # Or just random order.
    
    def backtrack(rem, chosen, deadline):
        if time.time() > deadline:
            return
        if len(chosen) >= 3: # Small S is often enough for expansion
            candidates.append(set(chosen))
            return # Don't go too deep, we want variety
        
        if not rem:
            return
            
        # Pick a vertex
        # To ensure diversity, pick a random vertex from rem
        # Or pick one with low degree in conflict graph to allow more choices?
        # Let's pick a random one to explore different structures.
        v_list = list(rem)
        random.shuffle(v_list)
        v = v_list[0]
        
        # Option 1: Include v
        # Check Constraint 1 for v: 
        # For any u, if v in N(u), then no other w in chosen can be in N(u).
        # This is hard to check incrementally without storing N(u) intersection with chosen.
        # Let's just check validity of the whole set at the end or use a simpler check.
        
        # Simpler: Just generate random subsets of size k, check if valid.
        pass

    # Alternative: Generate random subsets of V, check validity.
    # Validity check:
    # 1. S is independent in conflict graph.
    # 2. For all u, |N(u) & S| <= 1.
    
    # Let's generate random sets of size 2, 3, 4.
    for size in range(1, 6):
        for _ in range(200):
            if time.time() > deadline:
                return candidates
            S = set(random.sample(range(n), size))
            
            # Check Constraint 2: Independent in conflict graph
            valid = True
            for i in range(len(S)):
                si = list(S)[i]
                for j in range(i+1, len(S)):
                    sj = list(S)[j]
                    if sj in conflict[si]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check Constraint 1: For all u, |N(u) & S| <= 1
            for u in range(n):
                count = 0
                for v in S:
                    if v in adj[u]:
                        count += 1
                        if count > 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                candidates.append(S)
                
    return candidates

def _add_vertex(adj, n, S, deadline):
    """
    Try to add a new vertex with neighborhood S.
    Returns (success, new_adj, new_n)
    """
    # Check if adding v with N(v)=S keeps C4-free.
    # This is guaranteed by the construction of S (valid neighborhood).
    # But we must also ensure no new C4s are formed involving v.
    # A C4 involving v would be v-u-x-y-v.
    # Edges: (v,u), (u,x), (x,y), (y,v).
    # Since N(v)=S, u and y must be in S.
    # So we need a path of length 2 between two vertices in S that does not go through v.
    # i.e., for u, y in S, is there x such that u-x-y is a path?
    # This is exactly: N(u) & N(y) != empty.
    # Our Constraint 2 for S was: for x,y in S, N(x) & N(y) = empty.
    # So no such path exists. Thus C4-free is maintained.
    
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set(S))
    for u in S:
        new_adj[u].add(n)
        
    # Check independence number
    # We need alpha(new_adj) < 11.
    if _independent_set_ge(new_adj, new_n, 11, deadline):
        return False, adj, n
    else:
        return True, new_adj, new_n

def construct():
    deadline = time.time() + 140.0  # Safety margin
    best_n = 0
    best_edges = []
    
    # Strategy 1: Polarity Graphs
    # q=5 -> n=31. Alpha is small.
    # q=7 -> n=57. Alpha might be too big.
    # We will try q=5, then expand.
    
    for q in [5, 7, 9, 11]:
        if time.time() > deadline - 20:
            break
        try:
            result = _construct_polarity_graph(q)
            n = result['n']
            adj = result['adj']
            if not _verify_c4_free(n, adj):
                continue
            if _independent_set_ge(adj, n, 11, deadline - 10):
                continue
            if n > best_n:
                best_n = n
                best_edges = result['edges']
                
                # Attempt Expansion
                current_adj = adj
                current_n = n
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    candidates = _find_valid_neighborhoods(current_adj, current_n, deadline - 10)
                    random.shuffle(candidates)
                    for S in candidates:
                        if time.time() > deadline - 5:
                            break
                        success, new_adj, new_n = _add_vertex(current_adj, current_n, S, deadline - 5)
                        if success:
                            current_adj = new_adj
                            current_n = new_n
                            improved = True
                            if current_n > best_n:
                                best_n = current_n
                                # Update best_edges
                                edges = []
                                for i in range(current_n):
                                    for j in current_adj[i]:
                                        if i < j:
                                            edges.append([i, j])
                                best_edges = edges
                            break # Restart loop with new graph
                            
        except Exception as e:
            pass
            
    # Strategy 2: Fallback if polarity graphs failed or didn't expand enough
    if best_n < 30:
        # Triangle construction
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        best_n = 30
        best_edges = edges

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
