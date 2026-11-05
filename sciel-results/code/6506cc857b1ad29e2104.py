import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    def get_paley(q):
        """Generate Paley graph on q vertices (q must be prime ≡ 1 mod 4)."""
        if q % 4 != 1:
            return None
        residues = {(x * x) % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return q, adj
    
    def has_k5(n, adj):
        """Check if graph has K5 using bitset operations."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Check if common neighbors contain a triangle
                    if has_k3_in_mask(common, adj):
                        return True
        return False
    
    def has_k3_in_mask(mask, adj):
        """Check if induced subgraph on vertices in mask contains a triangle."""
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            nx = adj[x] & m
            if nx:
                # Check if any pair in nx are adjacent
                ny = nx
                while ny:
                    lsb2 = ny & -ny
                    y = lsb2.bit_length() - 1
                    ny ^= lsb2
                    if (adj[y] & nx) & ~(1 << y):
                        return True
        return False
    
    def find_k5(n, adj):
        """Find a K5 in the graph, return the 5 vertices or None."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Find triangle in common
                    m = common
                    while m:
                        lsb = m & -m
                        x = lsb.bit_length() - 1
                        m ^= lsb
                        nx = adj[x] & m
                        if nx:
                            ny = nx
                            while ny:
                                lsb2 = ny & -ny
                                y = lsb2.bit_length() - 1
                                ny ^= lsb2
                                if (adj[y] & nx) & ~(1 << y):
                                    # Found triangle x, y, z where z is in nx & adj[y]
                                    z_mask = (adj[y] & nx) & ~(1 << y)
                                    if z_mask:
                                        z = (z_mask & -z_mask).bit_length() - 1
                                        return [u, v, x, y, z]
        return None
    
    def find_i5(n, adj):
        """Find an independent set of size 5 (K5 in complement)."""
        full_mask = (1 << n) - 1
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
        result = find_k5(n, non_adj)
        return result
    
    def is_valid(n, adj):
        """Check if graph is (5,5)-free."""
        if find_k5(n, adj) is not None:
            return False
        if find_i5(n, adj) is not None:
            return False
        return True
    
    def count_violations(n, adj):
        """Count number of K5 and I5 violations (for heuristic)."""
        count = 0
        # Count K5
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    if has_k3_in_mask(common, adj):
                        count += 1
        # Count I5
        full_mask = (1 << n) - 1
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
        for u in range(n):
            for v in range(u + 1, n):
                if (non_adj[u] >> v) & 1:
                    common = non_adj[u] & non_adj[v]
                    if has_k3_in_mask(common, non_adj):
                        count += 1
        return count
    
    def local_search(n, adj, deadline):
        """Local search to find (5,5)-free graph."""
        current_adj = adj[:]
        best_adj = current_adj[:]
        best_score = -1
        
        # Try to find a valid graph
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Check if valid
            k5 = find_k5(n, current_adj)
            if k5 is not None:
                # Flip an edge in the K5 to break it
                vertices = k5
                pairs = list(itertools.combinations(vertices, 2))
                random.shuffle(pairs)
                for u, v in pairs:
                    if u > v:
                        u, v = v, u
                    # Flip edge (u, v)
                    bit = 1 << v
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] &= ~bit
                        current_adj[v] &= ~(1 << u)
                    else:
                        current_adj[u] |= bit
                        current_adj[v] |= (1 << u)
                    break
            else:
                i5 = find_i5(n, current_adj)
                if i5 is not None:
                    vertices = i5
                    pairs = list(itertools.combinations(vertices, 2))
                    random.shuffle(pairs)
                    for u, v in pairs:
                        if u > v:
                            u, v = v, u
                        # Flip edge (u, v)
                        bit = 1 << v
                        if (current_adj[u] >> v) & 1:
                            current_adj[u] &= ~bit
                            current_adj[v] &= ~(1 << u)
                        else:
                            current_adj[u] |= bit
                            current_adj[v] |= (1 << u)
                        break
                else:
                    # Valid!
                    return n, current_adj
            
        return None
    
    # Start with Paley 37
    p37 = get_paley(37)
    best_n, best_adj = 37, p37[1] if p37 else None
    
    # Try Paley 41
    p41 = get_paley(41)
    if p41:
        n41, adj41 = p41
        if is_valid(n41, adj41):
            best_n, best_adj = n41, adj41
    
    # Try to find larger graphs via local search
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline - 10:
            break
        if target_n <= best_n:
            continue
        
        time_budget = deadline - time.time()
        if time_budget < 10:
            break
        
        # Multiple restarts
        max_restarts = max(1, int(time_budget / 15))
        found = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(12345 + restart * 1000 + target_n)
            
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            result = local_search(target_n, adj, deadline)
            if result:
                found = result
                break
        
        if found:
            n_found, adj_found = found
            if n_found > best_n:
                best_n, best_adj = n_found, adj_found
                if best_n >= 42:
                    break
    
    # Convert to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges
