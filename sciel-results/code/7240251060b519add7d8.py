import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 18
    FULL = (1 << n) - 1

    # Precompute popcount for all masks
    POPC = [0] * (1 << n)
    for m in range(1, 1 << n):
        POPC[m] = POPC[m >> 1] + (m & 1)

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        # Check for K4
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                # Check if any pair in common (with index > b) are connected
                # We need c, d such that c > b, d > c, and c, d in common, and c-d edge exists.
                # Actually, just check if common contains an edge.
                # Let's iterate c in common with c > b
                mask_c = common & ~((1 << (b + 1)) - 1)
                while mask_c:
                    lsb = mask_c & -mask_c
                    c = lsb.bit_length() - 1
                    # Check if adj[c] & common has any bit > c
                    mask_d = adj[c] & common & ~((1 << (c + 1)) - 1)
                    if mask_d:
                        return True
                    mask_c ^= lsb
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    triangles.append((a, b, c))
                    common ^= lsb
        return triangles

    def max_triangle_free_size(triangles):
        """
        Find largest subset of vertices inducing no triangle.
        Branch and Bound.
        """
        if not triangles:
            return n
        
        # Map vertex to list of triangles it belongs to
        # And for each triangle, the 3 vertices
        # We need to know: if we pick v, which other vertices u become forbidden?
        # u is forbidden if there exists w in chosen such that {u, v, w} is a triangle.
        
        # Precompute: tri_neighbors[v] = set of u such that there is a triangle {u, v, w} for some w?
        # No, it depends on w.
        # If we have chosen set C, and we want to add v, we must remove any u in candidates
        # such that there exists w in C with {u, v, w} in triangles.
        
        # Let's precompute for each pair (u, v), the set of w such that {u, v, w} is a triangle.
        # pair_to_w[u][v] = set of w
        # This is expensive to store as sets of ints, but n=18 is small.
        # pair_w = [[set() for _ in range(n)] for _ in range(n)]
        
        # Actually, we can just check on the fly or precompute a bitmask.
        # mask_tri_uv[u][v] = bitmask of w such that {u, v, w} is a triangle.
        mask_tri = [[0] * n for _ in range(n)]
        for a, b, c in triangles:
            mask_tri[a][b] |= (1 << c)
            mask_tri[b][a] |= (1 << c)
            mask_tri[a][c] |= (1 << b)
            mask_tri[c][a] |= (1 << b)
            mask_tri[b][c] |= (1 << a)
            mask_tri[c][b] |= (1 << a)

        best = [0]
        
        # candidates: bitmask of vertices we can still add
        # chosen: bitmask of vertices we have added
        def dfs(candidates, chosen):
            if not candidates:
                if POPC[chosen] > best[0]:
                    best[0] = POPC[chosen]
                return
            
            # Upper bound
            if POPC[chosen] + POPC[candidates] <= best[0]:
                return
            
            # Heuristic: Pick a vertex with high degree in the "conflict" sense?
            # Or just the lowest index.
            # Let's pick a vertex v from candidates.
            # To prune better, pick v that is in the most triangles with other candidates?
            # For now, pick lowest set bit.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Branch 1: Exclude v
            new_cand_excl = candidates ^ lsb
            dfs(new_cand_excl, chosen)
            
            # Branch 2: Include v
            # New chosen: chosen | (1 << v)
            # New candidates: candidates without v, and without any u that forms a triangle with v and some w in chosen.
            # Forbidden u: u in candidates \ {v} such that mask_tri[u][v] & chosen != 0
            forbidden = 0
            cand_iter = new_cand_excl
            while cand_iter:
                lsb_u = cand_iter & -cand_iter
                u = lsb_u.bit_length() - 1
                if mask_tri[u][v] & chosen:
                    forbidden |= lsb_u
                cand_iter ^= lsb_u
            
            new_cand_incl = new_cand_excl & ~forbidden
            dfs(new_cand_incl, chosen | lsb)

        dfs(FULL, 0)
        return best[0]

    def remove_k4(adj):
        # Simple repair: remove edges from K4s
        for _ in range(100):
            if not has_k4(adj):
                return True
            # Find a K4
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1): continue
                    common = adj[a] & adj[b]
                    mask_c = common & ~((1 << (b + 1)) - 1)
                    while mask_c:
                        lsb_c = mask_c & -mask_c
                        c = lsb_c.bit_length() - 1
                        mask_d = adj[c] & common & ~((1 << (c + 1)) - 1)
                        if mask_d:
                            lsb_d = mask_d & -mask_d
                            d = lsb_d.bit_length() - 1
                            # Found K4 {a,b,c,d}
                            # Remove random edge
                            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e1, e2 = random.choice(edges)
                            adj[e1] &= ~(1 << e2)
                            adj[e2] &= ~(1 << e1)
                            found = True
                            break
                        mask_c ^= lsb_c
                    if found: break
                if found: break
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    
    best_adj = None
    best_score = 18
    
    # Initial candidates
    candidates = []
    
    # 1. Random graphs
    for _ in range(50):
        if time.time() > deadline - 60: break
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)
            
    # 2. Structured: Cyclic
    adj_s = [0] * n
    for i in range(n):
        for off in [1, 2, 4, 8]:
            j = (i + off) % n
            if j != i:
                adj_s[i] |= (1 << j)
                adj_s[j] |= (1 << i)
    if remove_k4(adj_s):
        candidates.append(adj_s)
        
    # 3. Structured: Paley-like (quadratic residues mod 19? No, n=18)
    # Just use a random regular graph
    for _ in range(10):
        if time.time() > deadline - 60: break
        adj = [0] * n
        # Generate random 7-regular graph
        deg = 7
        # Configuration model approximation
        # Simpler: random edges until degrees are close
        for i in range(n):
            adj[i] = 0
        # Shuffle edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        degs = [0] * n
        for u, v in edges:
            if degs[u] < deg and degs[v] < deg:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates
    for adj in candidates:
        if time.time() > deadline - 30: break
        triangles = get_triangles(adj)
        score = max_triangle_free_size(triangles)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        # Fallback
        best_adj = candidates[0]
        best_score = max_triangle_free_size(get_triangles(best_adj))

    # Local Search
    adj = best_adj[:]
    current_score = best_score
    temp = 1.0
    cooling = 0.99
    
    for i in range(10000):
        if time.time() > deadline - 5: break
        if best_score <= 7: break # 7 is likely optimal or near optimal
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        old_adj = adj[:]
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        triangles = get_triangles(adj)
        new_score = max_triangle_free_size(triangles)
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            adj = old_adj
            
        temp *= cooling
        if temp < 0.01: temp = 0.01

    return edges_from_adj(best_adj)
