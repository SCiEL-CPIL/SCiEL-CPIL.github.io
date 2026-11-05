"""SAT-based (DPLL) construction for R(C4,K11) > n."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(adj, n, limit=11):
    """Find an independent set of size >= limit using branch and bound.
    Returns True if an independent set of size >= limit exists."""
    remaining = set(range(n))
    
    def _branch(rem, current_size):
        if current_size >= limit:
            return True
        if not rem:
            return False
        if current_size + len(rem) < limit:
            return False
        # Pick a vertex with max degree in remaining (to prune)
        best_v = -1
        best_deg = -1
        for v in rem:
            deg_in_rem = len(adj[v] & rem)
            if deg_in_rem > best_deg:
                best_deg = deg_in_rem
                best_v = v
        # Branch 1: include best_v
        rem_without_neighbours = rem - adj[best_v] - {best_v}
        if _branch(rem_without_neighbours, current_size + 1):
            return True
        # Branch 2: exclude best_v
        if _branch(rem - {best_v}, current_size):
            return True
        return False
    
    return _branch(remaining, 0)

def _solve_c4_alpha(n, time_limit=10.0):
    """
    DPLL search for a C4-free graph on n vertices with alpha < 11.
    Returns (edges, adj) if found, else None.
    """
    start_time = time.time()
    sys.setrecursionlimit(10000)
    
    # adj[v] is a bitmask of neighbors
    adj = [0] * n
    # common[v][w] is a bitmask of common neighbors of v and w
    # We only need to track this for pairs where we might add edges
    # Actually, for C4-free, we need |N(u) & N(v)| <= 1.
    # This is equivalent to: for any u, v, the set of common neighbors has size <= 1.
    
    # To speed up, we can maintain a list of "forbidden" pairs or just check on the fly.
    # Checking on the fly with bitmasks is O(1) per pair check if we precompute common neighbors?
    # No, common neighbors change as we add edges.
    
    # Let's use a simpler representation for the search:
    # We build the graph vertex by vertex? Or edge by edge?
    # Edge by edge is harder to bound alpha.
    # Vertex by vertex: add vertex v, choose its neighbors S_v subset of {0..v-1}.
    # Constraints:
    # 1. C4-free: For any u, w < v, if u, w in S_v, then N(u) & N(w) must be empty (in the current graph).
    #    Also, for any u in S_v, and any w < v (w not in S_v), |N(u) & N(w)| <= 1 is already satisfied by previous steps?
    #    No, adding v adds v to N(u) for u in S_v.
    #    So for any pair (a, b) in the new graph, |N(a) & N(b)| <= 1.
    #    New pairs involving v: (v, u) for u in 0..v-1.
    #    |N(v) & N(u)| = |S_v & N(u)|. This must be <= 1 for all u.
    #    Pairs not involving v: (a, b) with a, b < v.
    #    N(a) and N(b) might have gained v as a neighbor if a, b in S_v.
    #    If a, b in S_v, then v is in N(a) and N(b).
    #    So |N(a) & N(b)| = |(N(a) & N(b))_old| + (1 if v in N(a) & N(b) else 0).
    #    Since v is new, v was not in N(a) or N(b) before.
    #    So if a, b in S_v, the common neighbor count increases by 1.
    #    Thus, if |(N(a) & N(b))_old| == 1, we CANNOT have both a, b in S_v.
    #    If |(N(a) & N(b))_old| >= 2, it was already invalid.
    
    # So the constraints for choosing S_v are:
    # 1. For all u < v: |S_v & N(u)| <= 1.
    # 2. For all pairs a, b < v: if |(N(a) & N(b))_old| == 1, then NOT (a in S_v AND b in S_v).
    #    This means S_v cannot contain any pair {a, b} that already has 1 common neighbor.
    #    Let C1 be the set of pairs {a, b} with exactly 1 common neighbor.
    #    S_v must be an independent set in the graph defined by C1 edges.
    
    # 3. Alpha < 11.
    #    This is the hard part to enforce incrementally.
    #    We can use a greedy check at the end, or prune during search.
    #    Pruning: If the current partial graph already has an independent set of size 11, prune.
    #    But checking alpha is expensive.
    #    We can use a bound: if the maximum possible alpha (given remaining vertices) is < 11, prune? No, we want alpha < 11.
    #    We want to AVOID alpha >= 11.
    #    So if we find an independent set of size 11, we prune.
    
    # Let's try a randomized greedy construction with backtracking on alpha violation?
    # Or just use the SAT solver logic for C4 and check alpha at the end?
    # If we just satisfy C4, the alpha might be large.
    # The polarity graph q=7 has n=57, alpha is likely > 10.
    # We need to enforce alpha < 11.
    
    # Strategy:
    # 1. Generate a C4-free graph using the vertex-addition method with random S_v.
    # 2. Check alpha. If alpha >= 11, backtrack or restart.
    # 3. To make it faster, we can try to keep degrees high to reduce alpha.
    
    def _random_c4_free_graph(n, seed):
        random.seed(seed)
        adj = [0] * n
        # common_count[a][b] = number of common neighbors of a and b
        # We can maintain this as a 2D array or dict of pairs.
        # For n=40, 2D array is 1600 ints, fine.
        cc = [[0] * n for _ in range(n)]
        
        for v in range(n):
            # Determine valid neighbors for v
            # Constraint 1: |S_v & N(u)| <= 1 for all u < v
            # Constraint 2: S_v is independent in C1 graph (pairs with cc[a][b]==1)
            
            # Let's collect candidates
            # A vertex u is a candidate if it doesn't violate constraints with other candidates?
            # It's a set packing problem.
            
            # Greedy approach:
            # Start with all u < v as candidates.
            # Iterate: pick a random candidate u, add to S_v, remove u and its "conflicts" from candidates.
            # Conflicts:
            # - If |N(u) & S_v| >= 1, then u cannot be added? No, |S_v & N(u)| <= 1.
            #   So if u is already in N(u') for some u' in S_v, then adding u makes |S_v & N(u)| = 1?
            #   Wait. The constraint is on the FINAL S_v.
            #   For each u, the number of neighbors of u in S_v must be <= 1.
            #   This is a matching-like constraint.
            
            # Let's just generate S_v randomly and check validity.
            # If invalid, retry or backtrack.
            
            max_tries = 100
            found = False
            for _ in range(max_tries):
                # Random subset of {0..v-1}
                # Size should be around n/4
                target_size = max(1, v // 4 + random.randint(-v//10, v//10))
                target_size = min(target_size, v)
                if target_size <= 0:
                    S_v = set()
                else:
                    candidates = list(range(v))
                    random.shuffle(candidates)
                    S_v = set(candidates[:target_size])
                
                # Check C4 constraints
                valid = True
                # 1. For each u in S_v, count neighbors in S_v
                #    Actually, for each u < v, count |S_v & N(u)|
                #    We can check for each pair a, b in S_v: are they both neighbors of some u?
                #    No, the condition is: for each u, |N(u) ∩ S_v| <= 1.
                #    This is equivalent to: no u has two neighbors in S_v.
                #    So for each u, if u is adjacent to a and b in S_v, invalid.
                
                # Faster check:
                # For each a in S_v, for each b in S_v (a < b):
                #    If there exists u such that u is neighbor of a and b, then u has 2 neighbors in S_v.
                #    So we need: for all pairs a, b in S_v, cc[a][b] == 0.
                #    Wait, cc[a][b] is the number of common neighbors of a and b in the CURRENT graph (before adding v).
                #    If cc[a][b] >= 1, then there is a u with u~a and u~b.
                #    If a, b in S_v, then v~a and v~b.
                #    Does this violate C4?
                #    C4 is a,b,u,v. Edges: a-b? No, C4 is a-u-b-v-a.
                #    Vertices: u, v, a, b.
                #    Edges: (u,a), (u,b), (v,a), (v,b).
                #    This is a C4 if u != v, a != b, etc.
                #    So if a, b in S_v, and there is a common neighbor u of a and b (u < v), then we have a C4.
                #    So the condition is: for all a, b in S_v, cc[a][b] == 0.
                
                # Also, we need to ensure that adding v doesn't create C4s involving v and two other vertices?
                # The C4 must involve v. So v is one of the 4 vertices.
                # The other 3 are u, a, b.
                # Edges: (v,a), (v,b), (u,a), (u,b).
                # This requires u to be a common neighbor of a and b.
                # So yes, the condition is exactly: S_v is a set of vertices with pairwise 0 common neighbors in the existing graph.
                
                # So S_v must be an independent set in the graph where edges connect pairs with cc[a][b] >= 1.
                
                # Let's check this condition.
                sv_list = list(S_v)
                for i in range(len(sv_list)):
                    a = sv_list[i]
                    for j in range(i+1, len(sv_list)):
                        b = sv_list[j]
                        if cc[a][b] > 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Update adj and cc
                    mask = 0
                    for u in S_v:
                        mask |= (1 << u)
                    adj[v] = mask
                    for u in S_v:
                        adj[u] |= (1 << v)
                        
                    # Update cc
                    # For each pair (x, y) in S_v: cc[x][y] += 1 (because v is a new common neighbor)
                    for u in S_v:
                        for w in S_v:
                            if u < w:
                                cc[u][w] += 1
                                cc[w][u] += 1
                    # For each u in S_v and w not in S_v (w < v):
                    # v is a neighbor of u, but not w.
                    # So common neighbors of u and w do not change?
                    # Wait. N(u) gains v. N(w) does not gain v.
                    # So N(u) & N(w) does not gain v.
                    # So cc[u][w] stays the same.
                    # What about pairs (u, v) where u < v?
                    # cc[u][v] = |N(u) & N(v)| = |N(u) & S_v|.
                    # We need to compute this.
                    # Since S_v has pairwise 0 common neighbors, for any u in S_v, |N(u) & S_v| = 0?
                    # No. u is in S_v. N(u) & S_v are neighbors of u that are also in S_v.
                    # Since S_v is independent in the cc>0 graph, there are no edges in the original graph between vertices in S_v?
                    # No, the graph is not necessarily triangle free.
                    # But if a, b in S_v, cc[a][b] = 0.
                    # This means they have no common neighbors.
                    # It does NOT mean they are not adjacent.
                    # If a~b, then b is in N(a).
                    # If b in S_v, then b is in N(a) & S_v.
                    # So cc[a][v] = |N(a) & S_v|.
                    # We can compute this by iterating.
                    
                    for u in S_v:
                        # cc[u][v] = |N(u) & S_v|
                        count = 0
                        for w in S_v:
                            if w != u and (adj[u] >> w) & 1:
                                count += 1
                        cc[u][v] = count
                        cc[v][u] = count
                        
                    found = True
                    break
            
            if not found:
                return None # Failed to add vertex
        
        # Convert to sets for alpha check
        adj_sets = [set() for _ in range(n)]
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                adj_sets[i].add(j)
                mask ^= lsb
        
        return adj_sets

    # Try to find a graph with n vertices
    # We will try random seeds
    # To ensure alpha < 11, we check after construction.
    # If alpha >= 11, we discard and try again.
    # This might be slow if most C4-free graphs have large alpha.
    # To bias towards small alpha, we should choose S_v to be as large as possible?
    # High degree -> small alpha (usually).
    
    best_adj = None
    best_edges = None
    
    # Try different n from high to low?
    # Or just try n=40, 39, etc.
    # The task asks to maximize n.
    # Let's try n=40 first. If it fails, 39, etc.
    
    for target_n in range(40, 25, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try a few random instances for this n
        for seed in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            adj_sets = _random_c4_free_graph(target_n, seed)
            if adj_sets is None:
                continue
            
            # Verify C4-free
            if not _verify_c4_free(target_n, adj_sets):
                continue
            
            # Check alpha
            if _max_independent_set(adj_sets, target_n, limit=11):
                continue
            
            # Valid!
            edges = []
            for i in range(target_n):
                for j in adj_sets[i]:
                    if i < j:
                        edges.append([i, j])
            
            return edges, adj_sets
        
    return None, None

def construct():
    deadline = time.time() + 145.0
    
    # Try SAT-based search
    edges, adj = _solve_c4_alpha(40, time_limit=120.0)
    
    if edges is not None:
        return {'n': 40, 'edges': edges}
    
    # Fallback: Polarity graph q=7 (n=57) is too big? No, n must be <= 43.
    # Polarity graph q=4 has n=21.
    # Polarity graph q=5 has n=31.
    # Polarity graph q=7 has n=57 (too big).
    # Polarity graph q=9 has n=91 (too big).
    
    # Let's try q=5 (n=31)
    # Or just return a small valid graph if search fails.
    
    # Fallback construction: Disjoint union of cliques?
    # C4-free: Cliques are C4-free.
    # Alpha: 1 per clique.
    # If we use K_10, alpha=1. n=10.
    # If we use 3 * K_10, n=30, alpha=3.
    # This is valid but small.
    
    # Better fallback: Polar graph q=4 (n=21)
    def _construct_polarity_graph(q):
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                    elif y != 0:
                        key = (0, 1, z * pow(y, q-2, q) % q)
                    else:
                        key = (0, 0, 1)
                    if key not in seen:
                        seen.add(key)
                        points.append(key)
        
        n_pts = len(points)
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                a1, b1, c1 = points[i]
                a2, b2, c2 = points[j]
                if (a1*a2 + b1*b2 + c1*c2) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        edges = []
        for i in range(n_pts):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n_pts, 'edges': edges, 'adj': adj}

    best_n = 0
    best_edges = []
    
    for q in [5, 4, 3]:
        if time.time() > deadline - 10:
            break
        try:
            result = _construct_polarity_graph(q)
            n = result['n']
            adj = result['adj']
            if _verify_c4_free(n, adj):
                if not _max_independent_set(adj, n, limit=11):
                    if n > best_n:
                        best_n = n
                        best_edges = result['edges']
        except Exception:
            continue
    
    if best_n == 0:
        # Last resort: 3 * K_10
        edges = []
        for t in range(3):
            offset = 10 * t
            for i in range(10):
                for j in range(i+1, 10):
                    edges.append([offset+i, offset+j])
        best_n = 30
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
