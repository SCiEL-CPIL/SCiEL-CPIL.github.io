import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    
    # GF(4) arithmetic
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        # 2*2=3, 2*3=1, 3*2=1, 3*3=2
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

    # Generate points of PG(2,4)
    points = []
    # Type 1: (1, a, b)
    for a in range(4):
        for b in range(4):
            points.append((1, a, b))
    # Type 2: (0, 1, c)
    for c in range(4):
        points.append((0, 1, c))
    # Type 3: (0, 0, 1)
    points.append((0, 0, 1))
    
    # Build adjacency for 21 points
    adj21 = [0] * 21
    for i in range(21):
        for j in range(i + 1, 21):
            p = points[i]
            q = points[j]
            # x1*x2 + y1*y2 + z1*z2 == 0
            val = gf4_add(gf4_add(gf4_mul(p[0], q[0]), gf4_mul(p[1], q[1])), gf4_mul(p[2], q[2]))
            if val == 0:
                adj21[i] |= (1 << j)
                adj21[j] |= (1 << i)
                
    # Remove vertex 20 (index 20) to get 20 vertices
    # We map indices 0..19 to the first 20 points
    adj = [0] * 20
    for i in range(20):
        for j in range(20):
            if i != j and (adj21[i] >> j) & 1:
                adj[i] |= (1 << j)
                
    def count_edges(adj):
        cnt = 0
        for i in range_n:
            cnt += bin(adj[i]).count('1')
        return cnt // 2

    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) creates a K3,3 subgraph.
        """
        # S_u = N(u) \ {v}
        # S_v = N(v) \ {u}
        s_u = adj[u] & ~(1 << v)
        s_v = adj[v] & ~(1 << u)
        
        # We need to check if there exist a, b in S_v and c, d in S_u
        # such that c, d are common neighbors of a and b.
        # i.e., (N(a) & N(b) & S_u) has at least 2 bits.
        
        mask_v = s_v
        while mask_v:
            lsb_a = mask_v & -mask_v
            a = lsb_a.bit_length() - 1
            mask_v ^= lsb_a
            
            # For this a, check all b in S_v \ {a}
            mask_b = s_v & ~(1 << a)
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                common_ab = adj[a] & adj[b]
                potential_cd = common_ab & s_u
                # If potential_cd has at least 2 bits, K3,3 exists
                if potential_cd & (potential_cd - 1):
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

    def get_missing_edges(adj):
        missing = []
        for u in range_n:
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        return missing

    def local_search(adj, max_iters=2000):
        """
        Local search to add edges.
        """
        current_count = count_edges(adj)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            missing = get_missing_edges(adj)
            if not missing:
                break
                
            random.shuffle(missing)
            
            # Try to add each missing edge
            for u, v in missing:
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    improved = True
                    # Restart search after improvement
                    break
        return current_count

    # Initial count
    init_count = count_edges(adj)
    
    # Perform local search
    # We might want to try removing some edges to allow adding more?
    # But the polarity graph is likely very close to optimal.
    # Let's just try adding edges first.
    local_search(adj)
    
    cur_count = count_edges(adj)
    
    # If we have time, try some random restarts or swaps
    # But adding edges is the primary goal.
    
    return adj_to_edges(adj)
