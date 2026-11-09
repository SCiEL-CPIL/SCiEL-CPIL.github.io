import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def build_masks(adj):
        n = adj.shape[0]
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    masks[i] |= (1 << j)
        return masks

    def has_k4(masks, n):
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            return True
        return False

    def has_ind6(adj, n):
        comp_masks = [0] * n
        for i in range(n):
            for j in range(n):
                if not adj[i, j] and i != j:
                    comp_masks[i] |= (1 << j)
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if not (comp_masks[wi] & (1 << xi)):
                            continue
                        common2 = common & comp_masks[wi] & comp_masks[xi]
                        common2 &= ~((1 << (xi + 1)) - 1)
                        y_mask = common2
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            if not (comp_masks[wi] & (1 << yi)) or not (comp_masks[xi] & (1 << yi)):
                                continue
                            common3 = common2 & comp_masks[yi]
                            common3 &= ~((1 << (yi + 1)) - 1)
                            z_mask = common3
                            while z_mask:
                                lsb_z = z_mask & -z_mask
                                zi = lsb_z.bit_length() - 1
                                z_mask ^= lsb_z
                                if (comp_masks[wi] & (1 << zi)) and (comp_masks[xi] & (1 << zi)) and (comp_masks[yi] & (1 << zi)):
                                    return True
        return False

    def is_valid(adj):
        n = adj.shape[0]
        masks = build_masks(adj)
        if has_k4(masks, n):
            return False
        if has_ind6(adj, n):
            return False
        return True

    def circulant_graph(n, connection_set):
        """Build circulant graph where vertex i connects to i+d mod n for d in connection_set."""
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for d in connection_set:
                j = (i + d) % n
                if j != i:
                    adj[i, j] = True
        return adj

    def count_violations_circulant(n, conn_mask):
        """Count K4 and independent 6-set violations for a circulant graph defined by connection mask.
        conn_mask is a bitmask where bit d is set if distance d is a connection."""
        # Build adjacency masks
        masks = [0] * n
        comp_masks = [0] * n
        for i in range(n):
            for d in range(1, n):
                j = (i + d) % n
                if conn_mask & (1 << d):
                    masks[i] |= (1 << j)
                else:
                    comp_masks[i] |= (1 << j)
        
        # Check K4
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1
                            if k4_count > 10:
                                return k4_count, 0
        if k4_count > 0:
            return k4_count, 0
        
        # Check independent 6-set
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if not (comp_masks[wi] & (1 << xi)):
                            continue
                        common2 = common & comp_masks[wi] & comp_masks[xi]
                        common2 &= ~((1 << (xi + 1)) - 1)
                        y_mask = common2
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            if not (comp_masks[wi] & (1 << yi)) or not (comp_masks[xi] & (1 << yi)):
                                continue
                            common3 = common2 & comp_masks[yi]
                            common3 &= ~((1 << (yi + 1)) - 1)
                            z_mask = common3
                            while z_mask:
                                lsb_z = z_mask & -z_mask
                                zi = lsb_z.bit_length() - 1
                                z_mask ^= lsb_z
                                if (comp_masks[wi] & (1 << zi)) and (comp_masks[xi] & (1 << zi)) and (comp_masks[yi] & (1 << zi)):
                                    ind6_count += 1
                                    if ind6_count > 10:
                                        return 0, ind6_count
        return 0, ind6_count

    def paley_graph(p):
        if p % 4 != 1:
            return None
        adj = np.zeros((p, p), dtype=bool)
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def local_search(adj, time_budget):
        n = adj.shape[0]
        end_time = time.time() + time_budget
        masks = build_masks(adj)
        
        if not has_k4(masks, n) and not has_ind6(adj, n):
            return adj, True
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old = adj[u, v]
            adj[u, v] = not old
            adj[v, u] = not old
            
            if old:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            
            if not has_k4(masks, n) and not has_ind6(adj, n):
                return adj, True
            
            if random.random() < 0.3:
                no_improve += 1
            else:
                adj[u, v] = old
                adj[v, u] = old
                if old:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                no_improve += 1
        
        if not has_k4(masks, n) and not has_ind6(adj, n):
            return adj, True
        return adj, False

    def spectral_circulant(n, num_trials=200):
        """Generate circulant graphs using spectral methods and return best valid one."""
        best_n_found = 0
        best_conn = None
        
        for trial in range(num_trials):
            if time_left() < 3.0:
                break
            
            # Generate random symmetric matrix
            A = np.random.randn(n, n)
            A = (A + A.T) / 2
            
            # Get eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
            except:
                continue
            
            # Use top k eigenvectors to create connection pattern
            k = random.randint(2, min(5, n // 2))
            top_vecs = eigvecs[:, -k:]
            
            # Create connection matrix from sign pattern
            conn_matrix = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Use projection onto top eigenvectors
                        proj = np.dot(top_vecs[i, :], top_vecs[j, :])
                        if proj > 0:
                            conn_matrix[i, j] = True
            
            # Extract circulant pattern (use first row as connection set)
            conn_mask = 0
            for d in range(1, n):
                if conn_matrix[0, d]:
                    conn_mask |= (1 << d)
            
            # Check if valid
            k4_c, ind6_c = count_violations_circulant(n, conn_mask)
            if k4_c == 0 and ind6_c == 0:
                return n, conn_mask
        
        return 0, None

    def search_circulant(n, time_budget):
        """Systematic search over circulant graphs of size n."""
        end_time = time.time() + time_budget
        half_n = n // 2
        max_conn = (1 << half_n) - 1  # only need first half due to symmetry
        
        # Start with random sampling
        for trial in range(5000):
            if time.time() > end_time:
                break
            if time_left() < 5.0:
                break
            
            conn_mask = random.randint(0, max_conn)
            # Ensure symmetry: if bit d is set, bit n-d should also be considered
            # For circulant graphs, connection at distance d implies connection at distance n-d
            # So we only need to specify distances 1..n/2
            
            k4_c, ind6_c = count_violations_circulant(n, conn_mask)
            if k4_c == 0 and ind6_c == 0:
                return conn_mask
        
        # If random didn't work, try structured patterns
        # Try all single-distance graphs
        for d in range(1, half_n + 1):
            if time.time() > end_time:
                break
            conn_mask = (1 << d) | (1 << (n - d)) if d != n - d else (1 << d)
            k4_c, ind6_c = count_violations_circulant(n, conn_mask)
            if k4_c == 0 and ind6_c == 0:
                return conn_mask
        
        # Try pairs of distances
        for d1 in range(1, half_n + 1):
            if time.time() > end_time:
                break
            for d2 in range(d1 + 1, half_n + 1):
                if time.time() > end_time:
                    break
                conn_mask = (1 << d1) | (1 << (n - d1)) | (1 << d2) | (1 << (n - d2))
                # Handle edge cases where d == n-d
                if d1 == n - d1:
                    conn_mask = (1 << d1) | (1 << d2) | (1 << (n - d2))
                if d2 == n - d2:
                    conn_mask = (1 << d1) | (1 << (n - d1)) | (1 << d2)
                k4_c, ind6_c = count_violations_circulant(n, conn_mask)
                if k4_c == 0 and ind6_c == 0:
                    return conn_mask
        
        return None

    # --- Main ---
    best_n = 17
    best_edges = None

    # Start with Paley(17)
    p17 = paley_graph(17)
    if p17 is not None:
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if p17[i, j]]

    # Try circulant graph search for larger n
    for target_n in [36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if time_left() < 10.0:
            break
        
        # Try spectral approach first
        if time_left() > 30.0:
            spec_n, spec_conn = spectral_circulant(target_n, num_trials=100)
            if spec_n == target_n and spec_conn is not None:
                adj = circulant_graph(target_n, [d for d in range(1, target_n) if spec_conn & (1 << d)])
                if is_valid(adj):
                    best_n = target_n
                    best_edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i, j]]
                    return {'n': best_n, 'edges': best_edges}
        
        # Try systematic circulant search
        budget = min(time_left() * 0.3, 30.0)
        conn_mask = search_circulant(target_n, budget)
        if conn_mask is not None:
            adj = circulant_graph(target_n, [d for d in range(1, target_n) if conn_mask & (1 << d)])
            if is_valid(adj):
                best_n = target_n
                best_edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i, j]]
                return {'n': best_n, 'edges': best_edges}
        
        # Try local search from random circulant
        if time_left() > 15.0:
            for trial in range(5):
                if time_left() < 5.0:
                    break
                conn_mask = random.randint(0, (1 << (target_n // 2)) - 1)
                adj = circulant_graph(target_n, [d for d in range(1, target_n) if conn_mask & (1 << d)])
                adj, found = local_search(adj, min(time_left() * 0.2, 15.0))
                if found:
                    best_n = target_n
                    best_edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i, j]]
                    return {'n': best_n, 'edges': best_edges}

    # Try known 35-vertex construction as fallback
    if time_left() > 10.0 and best_n < 35:
        adj35 = np.zeros((35, 35), dtype=bool)
        adj35[0:17, 0:17] = p17
        adj35[18:35, 18:35] = p17
        adj35[17, :] = True
        adj35[:, 17] = True
        adj35[17, 17] = False
        squares17 = set()
        for x in range(1, 17):
            squares17.add(x * x % 17)
        for i in range(17):
            for j in range(17):
                if (i + j) % 17 not in squares17:
                    adj35[i, 18 + j] = True
                    adj35[18 + j, i] = True
        
        if is_valid(adj35):
            best_n = 35
            best_edges = [[i, j] for i in range(35) for j in range(i + 1, 35) if adj35[i, j]]
            return {'n': best_n, 'edges': best_edges}
        
        adj35, found = local_search(adj35, min(time_left() * 0.4, 40.0))
        if found:
            best_n = 35
            best_edges = [[i, j] for i in range(35) for j in range(i + 1, 35) if adj35[i, j]]
            return {'n': best_n, 'edges': best_edges}

    # Try smaller n with random + local search
    for target_n in [30, 25, 20]:
        if time_left() < 5.0:
            break
        if best_n >= target_n:
            continue
        
        for trial in range(10):
            if time_left() < 3.0:
                break
            
            adj = np.zeros((target_n, target_n), dtype=bool)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.4:
                        adj[i, j] = True
                        adj[j, i] = True
            
            adj, found = local_search(adj, min(time_left() * 0.2, 15.0))
            if found:
                best_n = target_n
                best_edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i, j]]
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
