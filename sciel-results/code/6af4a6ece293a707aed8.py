import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute triples for faster K3,3 check
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    def get_common_neighbors(adj, u, v, w):
        # Returns list of common neighbors of u, v, w
        # Using bitmasks for speed? n=20 fits in int.
        # adj_mask[x] is an integer where bit y is 1 if x~y
        pass

    # Let's use bitmasks for adjacency to make intersection fast.
    # adj_mask[i] is a 20-bit integer.
    
    def check_k33(adj_mask):
        """
        Returns True if graph is K3,3 free.
        Also returns a violating triple and common neighbors if not.
        """
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            # Count bits
            if bin(common).count('1') >= 3:
                # Find 3 bits
                bits = [i for i in range(n) if (common >> i) & 1]
                return False, (u, v, w), bits[:3]
        return True, None, None

    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2

    best_edges = None
    best_count = 0

    for trial in range(200):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        # Phase 1: Make valid
        # While invalid, remove an edge from a K3,3
        valid, viol_uvw, viol_common = check_k33(adj_mask)
        while not valid:
            if time.time() > deadline:
                break
            # Remove a random edge between viol_uvw and viol_common
            # Pick random u from viol_uvw, random v from viol_common
            u = random.choice(viol_uvw)
            v = random.choice(viol_common)
            # Remove edge u-v
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            valid, viol_uvw, viol_common = check_k33(adj_mask)
            
        if time.time() > deadline:
            break
            
        # Phase 2: Maximize edges (Hill Climbing with Kicks)
        current_count = count_edges(adj_mask)
        
        # Try to add edges
        improved = True
        while improved:
            improved = False
            # Get list of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj_mask[i] >> j) & 1:
                        non_edges.append((i, j))
                        
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                # Try adding u-v
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
                is_valid, _, _ = check_k33(adj_mask)
                
                if is_valid:
                    current_count += 1
                    improved = True
                else:
                    # Revert
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
            if current_count > best_count:
                best_count = current_count
                # Store best state
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_mask[i] >> j) & 1:
                            best_edges.append([i, j])

        # Phase 3: Kick (Escape local optimum)
        # If we are stuck, remove a random edge, then fix and re-optimize
        # We do this a few times per trial
        for kick in range(5):
            if time.time() > deadline:
                break
                
            # Remove a random edge
            edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges_list.append((i, j))
                        
            if not edges_list:
                break
                
            u, v = random.choice(edges_list)
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            current_count -= 1
            
            # Make valid again
            valid, viol_uvw, viol_common = check_k33(adj_mask)
            while not valid:
                if time.time() > deadline:
                    break
                u_r = random.choice(viol_uvw)
                v_r = random.choice(viol_common)
                adj_mask[u_r] &= ~(1 << v_r)
                adj_mask[v_r] &= ~(1 << u_r)
                current_count -= 1
                valid, viol_uvw, viol_common = check_k33(adj_mask)
                
            if time.time() > deadline:
                break
                
            # Re-optimize
            improved = True
            while improved:
                improved = False
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not (adj_mask[i] >> j) & 1:
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                for u, v in non_edges:
                    if time.time() > deadline:
                        break
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    
                    is_valid, _, _ = check_k33(adj_mask)
                    
                    if is_valid:
                        current_count += 1
                        improved = True
                    else:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        
                if current_count > best_count:
                    best_count = current_count
                    best_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj_mask[i] >> j) & 1:
                                best_edges.append([i, j])

    if best_edges is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
