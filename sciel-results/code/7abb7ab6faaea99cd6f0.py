import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin (330s limit, 35s buffer)
    
    def time_ok():
        return time.time() < deadline
    
    def find_violation_fast(adj, n):
        """
        Fast check for K5 or independent 5-set using bitmasks.
        Returns (type, vertices) or None.
        """
        # Check for K5
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if au & (1 << v):
                    common_uv = au & adj[v]
                    # Need 3 more vertices in common_uv, each pair connected
                    w = 0
                    while common_uv:
                        w_bit = common_uv & -common_uv
                        w = w_bit.bit_length() - 1
                        common_uv ^= w_bit
                        if w <= v:
                            continue
                        common_uvw = common_uv & adj[w]
                        x = 0
                        while common_uvw:
                            x_bit = common_uvw & -common_uvw
                            x = x_bit.bit_length() - 1
                            common_uvw ^= x_bit
                            if x <= w:
                                continue
                            common_uvwx = common_uvw & adj[x]
                            if common_uvwx:
                                y_bit = common_uvwx & -common_uvwx
                                y = y_bit.bit_length() - 1
                                return ('clique', [u, v, w, x, y])
        
        # Check for Independent Set of size 5
        # non_adj[i] = all j != i not connected to i
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
        
        for u in range(n):
            nu = non_adj[u]
            for v in range(u + 1, n):
                if nu & (1 << v):
                    common_uv = nu & non_adj[v]
                    w = 0
                    while common_uv:
                        w_bit = common_uv & -common_uv
                        w = w_bit.bit_length() - 1
                        common_uv ^= w_bit
                        if w <= v:
                            continue
                        common_uvw = common_uv & non_adj[w]
                        x = 0
                        while common_uvw:
                            x_bit = common_uvw & -common_uvw
                            x = x_bit.bit_length() - 1
                            common_uvw ^= x_bit
                            if x <= w:
                                continue
                            common_uvwx = common_uvw & non_adj[x]
                            if common_uvwx:
                                y_bit = common_uvwx & -common_uvwx
                                y = y_bit.bit_length() - 1
                                return ('independent', [u, v, w, x, y])
        
        return None

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    best_n = q
    best_adj = adj[:]
    
    def try_grow(current_adj, current_n, target_n, max_time=5.0):
        """
        Try to grow graph from current_n to target_n vertices.
        Returns (new_adj, success) or (None, False).
        """
        start = time.time()
        
        for attempt in range(100):
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            # Create new graph with random connections for new vertex
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = current_adj[i]
            
            new_idx = target_n - 1
            # Random connection pattern
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            # Local search to fix violations
            local_deadline = time.time() + 3.0
            valid = False
            stuck_count = 0
            last_viol = None
            
            while time.time() < local_deadline:
                viol = find_violation_fast(new_adj, target_n)
                if viol is None:
                    valid = True
                    break
                
                vtype, verts = viol
                
                # Check if stuck (same violation repeatedly)
                if last_viol is not None and last_viol == (vtype, tuple(sorted(verts))):
                    stuck_count += 1
                    if stuck_count > 20:
                        # Try a different random pair from the violating set
                        u, v = random.sample(verts, 2)
                        if vtype == 'clique':
                            new_adj[u] &= ~(1 << v)
                            new_adj[v] &= ~(1 << u)
                        else:
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                        stuck_count = 0
                        last_viol = None
                        continue
                else:
                    stuck_count = 0
                
                last_viol = (vtype, tuple(sorted(verts)))
                
                # Pick a pair to fix
                if vtype == 'clique':
                    # Remove an edge from the clique
                    # Try to pick an edge that minimizes damage
                    u, v = random.sample(verts, 2)
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Add an edge to break independence
                    u, v = random.sample(verts, 2)
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
            
            if valid:
                return new_adj, True
        
        return None, False
    
    # Try to grow step by step
    for target_n in range(current_n + 1, 47):
        if not time_ok():
            break
        
        success = False
        # Try multiple times with different random seeds
        for trial in range(3):
            if not time_ok():
                break
            
            new_adj, ok = try_grow(adj, current_n, target_n, max_time=8.0)
            if ok:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                success = True
                break
        
        if not success:
            # Try one more time with a different approach: 
            # add vertex with all edges, then remove to fix cliques
            if time_ok():
                start = time.time()
                for attempt in range(20):
                    if time.time() - start > 10.0 or time.time() > deadline - 15:
                        break
                    
                    new_adj = [0] * target_n
                    for i in range(current_n):
                        new_adj[i] = adj[i]
                    
                    new_idx = target_n - 1
                    # Start with all edges to new vertex
                    for i in range(current_n):
                        new_adj[i] |= (1 << new_idx)
                        new_adj[new_idx] |= (1 << i)
                    
                    # Local search
                    local_deadline = time.time() + 3.0
                    valid = False
                    while time.time() < local_deadline:
                        viol = find_violation_fast(new_adj, target_n)
                        if viol is None:
                            valid = True
                            break
                        
                        vtype, verts = viol
                        if vtype == 'clique':
                            u, v = random.sample(verts, 2)
                            new_adj[u] &= ~(1 << v)
                            new_adj[v] &= ~(1 << u)
                        else:
                            u, v = random.sample(verts, 2)
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                    
                    if valid:
                        current_n = target_n
                        adj = new_adj
                        best_n = current_n
                        best_adj = adj[:]
                        success = True
                        break
            
            if not success:
                # Try another approach: start with no edges to new vertex
                if time_ok():
                    start = time.time()
                    for attempt in range(20):
                        if time.time() - start > 10.0 or time.time() > deadline - 15:
                            break
                        
                        new_adj = [0] * target_n
                        for i in range(current_n):
                            new_adj[i] = adj[i]
                        
                        new_idx = target_n - 1
                        # Start with no edges to new vertex
                        # (new_adj[new_idx] is already 0)
                        
                        # Local search
                        local_deadline = time.time() + 3.0
                        valid = False
                        while time.time() < local_deadline:
                            viol = find_violation_fast(new_adj, target_n)
                            if viol is None:
                                valid = True
                                break
                            
                            vtype, verts = viol
                            if vtype == 'clique':
                                u, v = random.sample(verts, 2)
                                new_adj[u] &= ~(1 << v)
                                new_adj[v] &= ~(1 << u)
                            else:
                                u, v = random.sample(verts, 2)
                                new_adj[u] |= (1 << v)
                                new_adj[v] |= (1 << u)
                        
                        if valid:
                            current_n = target_n
                            adj = new_adj
                            best_n = current_n
                            best_adj = adj[:]
                            success = True
                            break
            
            if not success:
                break
    
    # Final verification
    if find_violation_fast(best_adj, best_n) is not None:
        # Fallback to Paley
        best_n = 37
        best_adj = [0] * 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)

    return best_n, get_edges(best_adj, best_n)
