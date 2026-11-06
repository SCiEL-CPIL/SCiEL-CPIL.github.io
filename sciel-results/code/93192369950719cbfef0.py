import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120 s evaluator kill

    def is_c4_free(adj, n):
        # C4-free iff no two vertices have >= 2 common neighbors.
        # adj: list of Python sets
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if v > u:
                    if len(nu & adj[v]) >= 2:
                        return False
        return True

    def has_independent_set_11(adj, n):
        # Exact decision: does there exist an independent set of size 11?
        # Backtracking with bitmasks for speed (n <= 43).
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_masks[u] = mask

        # Order vertices by degree (higher degree first) for better pruning
        order = list(range(n))
        order.sort(key=lambda x: bin(adj_masks[x]).count('1'), reverse=True)
        # Map old index -> position in order, and precompute masks in order
        pos_of = [0] * n
        ordered_adj_masks = [0] * n
        for new_idx, old_idx in enumerate(order):
            pos_of[old_idx] = new_idx
            # Convert mask to ordered representation
            m = 0
            for v in adj[old_idx]:
                m |= (1 << pos_of[v])
            ordered_adj_masks[new_idx] = m

        target = 11
        full_mask = (1 << n) - 1

        def dfs(available, size):
            if size + bin(available).count('1') < target:
                return False
            if size == target:
                return True
            # Pick a vertex from available with max degree in available
            # Find first set bit (heuristic: just pick first for speed, but max degree is better)
            # For n<=43, iterating is fine.
            verts = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            if not verts:
                return False
            
            # Sort by degree in available (descending)
            verts.sort(key=lambda v: bin(ordered_adj_masks[v] & available).count('1'), reverse=True)
            v = verts[0]
            
            # Branch 1: Include v
            # Remove v and its neighbors
            new_available = available & ~(1 << v) & ~ordered_adj_masks[v]
            if dfs(new_available, size + 1):
                return True
            
            # Branch 2: Exclude v
            new_available = available & ~(1 << v)
            if dfs(new_available, size):
                return True
            
            return False

        return dfs(full_mask, 0)

    def get_polarity_graph_q7():
        # Construct polarity graph of PG(2,7).
        # Points: 43. Lines: 43.
        # Polarity: (x,y,z) is orthogonal to (x',y',z') if xx' + yy' + zz' == 0 mod 8.
        # In PG(2,q), points are 1-dim subspaces of F_q^3.
        # We enumerate all non-zero triples (x,y,z) in F_8^3, identify projective points.
        # F_8: GF(8) with primitive element alpha, alpha^3 + alpha + 1 = 0.
        # Elements: 0, 1, a, a^2, a^3, a^4, a^5, a^6.
        # a^3 = a + 1
        # a^4 = a^2 + a
        # a^5 = a^2 + a + 1
        # a^6 = a^2 + 1
        # a^7 = 1
        
        # We can represent elements as integers 0..7.
        # Multiplication table for F_8.
        # Let's build it.
        # 0: 0
        # 1: 1
        # a: a
        # a^2: a^2
        # a^3: a+1
        # a^4: a^2+a
        # a^5: a^2+a+1
        # a^6: a^2+1
        
        # Addition is XOR in basis (1, a, a^2) if we represent elements as bitmasks?
        # Let's represent elements as 3-bit integers: bit0=1, bit1=a, bit2=a^2.
        # 0: 000
        # 1: 001
        # a: 010
        # a^2: 100
        # a+1: 011
        # a^2+a: 110
        # a^2+a+1: 111
        # a^2+1: 101
        
        # So mapping:
        # 0 -> 0
        # 1 -> 1
        # a -> 2
        # a^2 -> 4
        # a^3 (a+1) -> 3
        # a^4 (a^2+a) -> 6
        # a^5 (a^2+a+1) -> 7
        # a^6 (a^2+1) -> 5
        
        # Multiplication:
        # We can compute a^i * a^j = a^(i+j mod 7) (for non-zero).
        # Let's precompute mul_table[8][8].
        
        def build_mul_table():
            # Map exponent to bitmask
            exp_to_mask = [0, 1, 2, 4, 3, 6, 7, 5]
            # Map mask to exponent
            mask_to_exp = [0] * 8
            for e in range(8):
                mask_to_exp[exp_to_mask[e]] = e
                
            table = [[0]*8 for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    if i == 0 or j == 0:
                        table[i][j] = 0
                    else:
                        exp_i = mask_to_exp[i]
                        exp_j = mask_to_exp[j]
                        exp_res = (exp_i + exp_j) % 7
                        table[i][j] = exp_to_mask[exp_res]
            return table

        mul_table = build_mul_table()
        
        def add_f8(x, y):
            return x ^ y # Since basis is (1, a, a^2) and char 2
        
        def dot_product(u, v):
            # u, v are tuples of 3 F_8 elements (bitmasks)
            s = 0
            for i in range(3):
                s = add_f8(s, mul_table[u[i]][v[i]])
            return s

        # Enumerate points
        points = []
        seen = set()
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero element 1
                    # Find first non-zero
                    first_non_zero = -1
                    for i in range(3):
                        if (x >> i) & 1 != 0 or (y >> i) & 1 != 0 or (z >> i) & 1 != 0:
                            # This is not how we store. We store x,y,z as F8 elements.
                            pass
                    
                    # Let's just store (x,y,z) and normalize.
                    # Normalize: find first non-zero coordinate.
                    coords = [x, y, z]
                    first_idx = -1
                    for i in range(3):
                        if coords[i] != 0:
                            first_idx = i
                            break
                    if first_idx == -1:
                        continue
                    inv = 1 # In F8, inverse of non-zero is a^(6-exp)
                    # Find inverse of coords[first_idx]
                    exp = mask_to_exp[coords[first_idx]]
                    inv_exp = (7 - exp) % 7
                    inv_val = exp_to_mask[inv_exp]
                    
                    norm_coords = [0, 0, 0]
                    for i in range(3):
                        norm_coords[i] = mul_table[coords[i]][inv_val]
                    
                    key = tuple(norm_coords)
                    if key not in seen:
                        seen.add(key)
                        points.append(key)
                        
        # There should be 43 points.
        n_pts = len(points)
        if n_pts != 43:
            # Fallback if construction fails
            return None

        # Build adjacency
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if dot_product(points[i], points[j]) == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # Fallback: 10 disjoint triangles
    fallback_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        fallback_edges += [[a,b], [a,c], [b,c]]

    # Try to find a valid subgraph of PG(2,7) polarity graph
    pg_adj = get_polarity_graph_q7()
    
    if pg_adj is None:
        return {'n': 30, 'edges': fallback_edges}

    n_total = 43
    # We want to find a subset of vertices of size n such that the induced subgraph is C4-free and alpha <= 10.
    # The full graph is C4-free (polarity graph of projective plane).
    # So any induced subgraph is C4-free.
    # We just need to check alpha <= 10.
    # We want to maximize n.
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try n=43 down to 31
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Randomly sample subsets of size n?
        # Or use a heuristic: remove vertices with high degree in independent set?
        # Actually, we can just try random subsets.
        # But 43 choose 43 is 1. 43 choose 42 is 43. 43 choose 41 is ~900.
        # So we can try all subsets for n close to 43.
        
        trials = 0
        max_trials = 1000
        found = False
        
        while trials < max_trials and time.time() < deadline - 5:
            trials += 1
            if n == 43:
                subset = list(range(43))
            elif n == 42:
                subset = list(range(43))
                subset.remove(random.randint(0, 42))
            else:
                subset = random.sample(range(43), n)
            
            # Check alpha of induced subgraph
            # Map subset indices to 0..n-1
            idx_map = {old: new for new, old in enumerate(subset)}
            adj_sub = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if subset[j] in pg_adj[subset[i]]:
                        adj_sub[i].add(j)
                        adj_sub[j].add(i)
                        
            if not has_independent_set_11(adj_sub, n):
                # Valid!
                best_n = n
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if j in adj_sub[i]:
                            best_edges.append([i, j])
                found = True
                break
            
            # If n is small, we can try more trials.
            # If n is large, trials are few.
            
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
