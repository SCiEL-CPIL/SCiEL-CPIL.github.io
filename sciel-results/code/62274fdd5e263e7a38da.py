import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) creates a K3,3 subgraph.
        Pure bitmask operations.
        """
        c_y_mask = adj[u] & ~(1 << v)
        c_x_mask = adj[v] & ~(1 << u)
        
        # Need at least 2 bits in c_y_mask
        if not (c_y_mask & (c_y_mask - 1)):
            return True
        
        # Iterate over pairs in c_y_mask using bit tricks
        mask = c_y_mask
        while mask:
            lsb1 = mask & -mask
            y1 = lsb1.bit_length() - 1
            mask2 = mask ^ lsb1
            mask = mask2
            while mask2:
                lsb2 = mask2 & -mask2
                y2 = lsb2.bit_length() - 1
                mask2 = mask2 ^ lsb2
                
                potential_x = (adj[y1] & adj[y2]) & c_x_mask
                if potential_x & (potential_x - 1):
                    return False
        return True

    def adj_to_edges(adj):
        edges = []
        for i in range_n:
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def count_edges(adj):
        cnt = 0
        for i in range_n:
            cnt += bin(adj[i]).count('1')
        return cnt // 2

    def get_missing_edges(adj):
        missing = []
        for u in range_n:
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        return missing

    def build_polarity_graph(vertex_order=None):
        """Construct a K3,3-free graph from polarity of projective plane of order 4."""
        def gf4_add(a, b):
            return a ^ b
        
        def gf4_mul(a, b):
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1:
                return a
            lookup = {(2,2): 3, (2,3): 1, (3,2): 1, (3,3): 2}
            return lookup[(a, b)]
        
        def gf4_inv(a):
            if a == 0:
                return 0
            if a == 1:
                return 1
            if a == 2:
                return 3
            if a == 3:
                return 2
        
        points = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = gf4_inv(x)
                        pt = (1, gf4_mul(inv_x, y), gf4_mul(inv_x, z))
                    elif y != 0:
                        inv_y = gf4_inv(y)
                        pt = (0, 1, gf4_mul(inv_y, z))
                    else:
                        pt = (0, 0, 1)
                    
                    found = False
                    for p in points:
                        if p == pt:
                            found = True
                            break
                    if not found:
                        points.append(pt)
        
        def is_orthogonal(p, q):
            val = gf4_add(gf4_add(gf4_mul(p[0], q[2]), gf4_mul(p[2], q[0])), gf4_mul(p[1], q[1]))
            return val == 0
        
        adj21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if is_orthogonal(points[i], points[j]):
                    adj21[i] |= (1 << j)
                    adj21[j] |= (1 << i)
        
        # Use first 20 points
        adj20 = [0] * 20
        for i in range(20):
            for j in range(20):
                if i != j and (adj21[i] >> j) & 1:
                    adj20[i] |= (1 << j)
        
        return adj20

    def is_k33_free(adj):
        for i in range_n:
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if bin(common).count('1') >= 3:
                        return False
        return True

    best_edges = []
    best_count = 0
    best_adj = [0] * n

    # Build initial polarity graph
    init_adj = build_polarity_graph()
    
    if is_k33_free(init_adj):
        init_count = count_edges(init_adj)
        if init_count > best_count:
            best_count = init_count
            best_adj = init_adj.copy()
            best_edges = adj_to_edges(init_adj)

    # Local search: fast edge-swap refinement
    # Strategy: remove one edge, try to add back a different edge that involves
    # vertices near the removed edge. Much faster than trying all missing edges.
    
    def local_search(adj, max_iters=5000):
        """Fast local search using targeted edge swaps."""
        current_count = count_edges(adj)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            if current_count == 0:
                break
            
            # Get current edges for random selection
            edges = []
            for i in range_n:
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append((i, j))
                    mask ^= lsb
            
            if not edges:
                break
            
            # Pick a random edge to remove
            edge_idx = random.randint(0, len(edges) - 1)
            u_rem, v_rem = edges[edge_idx]
            
            # Remove it
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_count -= 1
            
            # Build candidate edges to try adding back:
            # 1. The original edge (to revert if nothing better)
            # 2. Edges between neighbors of u_rem and neighbors of v_rem
            # 3. Random missing edges
            candidates = []
            
            # Neighbors of u_rem (excluding v_rem)
            neu = [i for i in range_n if i != u_rem and i != v_rem and (adj[u_rem] >> i) & 1]
            # Neighbors of v_rem (excluding u_rem)
            nev = [i for i in range_n if i != u_rem and i != v_rem and (adj[v_rem] >> i) & 1]
            
            # Try edges between neu and nev (cross edges)
            for a in neu:
                for b in nev:
                    if a != b and not ((adj[a] >> b) & 1):
                        candidates.append((min(a,b), max(a,b)))
            
            # Also try edges from u_rem to nev and v_rem to neu
            for b in nev:
                if not ((adj[u_rem] >> b) & 1):
                    candidates.append((min(u_rem,b), max(u_rem,b)))
            for a in neu:
                if not ((adj[v_rem] >> a) & 1):
                    candidates.append((min(v_rem,a), max(v_rem,a)))
            
            # Add some random missing edges
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for e in missing[:10]:
                if e not in candidates:
                    candidates.append(e)
            
            random.shuffle(candidates)
            
            added = False
            for u, v in candidates:
                if (adj[u] >> v) & 1:
                    continue  # already present
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_addition(adj, u, v):
                    current_count += 1
                    added = True
                    break
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if added:
                improved = True
            else:
                # Revert
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                current_count += 1
        
        return current_count

    # Main search loop
    while time.time() < deadline:
        # Start from best or random
        if random.random() < 0.6 and best_count > 0:
            adj = best_adj.copy()
        else:
            # Random greedy construction
            shuffled = all_edges.copy()
            random.shuffle(shuffled)
            adj = [0] * n
            for u, v in shuffled:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_addition(adj, u, v):
                    pass  # keep
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Local search refinement
        local_search(adj)
        
        cur_count = count_edges(adj)
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj.copy()
            best_edges = adj_to_edges(adj)

    return best_edges
