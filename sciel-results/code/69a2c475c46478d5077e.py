import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_n = 17
    best_edges = None
    
    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17
    best_adj = np.zeros((17, 17), dtype=bool)
    for u, v in edges_17:
        best_adj[u, v] = True
        best_adj[v, u] = True

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def has_k4(n, masks):
        """Check if any K4 exists. Early exit."""
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

    def has_ind6(n, masks):
        """Check if any independent 6-set exists. Early exit."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
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
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        return True
        return False

    def is_valid(n, adj):
        """Check validity with early exit."""
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def count_violations(n, adj):
        """Count total K4 + Ind6 violations."""
        masks = build_masks(n, adj)
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
        
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
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
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1
        return k4_count + ind6_count

    def spectral_generate(n, seed=None):
        """Generate candidate graph using spectral methods."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate random symmetric matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2.0
        
        # Add structure: make it more likely to have good properties
        # Use a combination of random and structured components
        for _ in range(3):
            B = np.random.randn(n, n)
            B = (B + B.T) / 2.0
            A += 0.5 * B
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        # Take top k eigenvectors and binarize
        # Try different numbers of eigenvectors
        best_adj = None
        best_score = float('inf')
        
        for k in [2, 3, 4, 5]:
            if k >= n:
                continue
            V = eigenvectors[:, :k]
            # Project onto first eigenvector primarily
            scores = V[:, 0] * 2.0 + V[:, 1] * 1.0 + (V[:, 2] * 0.5 if k > 2 else 0)
            
            # Binarize: edge if both endpoints have positive score
            threshold = np.median(scores)
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if scores[i] > threshold and scores[j] > threshold:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Also try: edge if scores are close (small difference)
            adj2 = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(scores[i] - scores[j]) < np.std(scores):
                        adj2[i, j] = True
                        adj2[j, i] = True
            
            # Check both and keep the better one
            for candidate in [adj, adj2]:
                score = count_violations(n, candidate)
                if score < best_score:
                    best_score = score
                    best_adj = candidate.copy()
        
        return best_adj

    def local_search(n, adj, time_budget):
        """Local search with bitmask updates."""
        end_time = time.time() + time_budget
        masks = build_masks(n, adj)
        current_cost = count_violations(n, adj)
        T = 3.0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_state = adj[u, v]
            adj[u, v] = not old_state
            adj[v, u] = not old_state
            
            # Update masks
            if old_state:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            
            new_cost = count_violations(n, adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost == 0:
                    return adj, True
            else:
                delta = new_cost - current_cost
                T = max(0.5, T * 0.995)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Revert
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    if old_state:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                    no_improve += 1
        
        return adj, (current_cost == 0)

    def try_find_graph(n, time_budget):
        """Try multiple spectral generations + local search."""
        end_time = time.time() + time_budget
        
        best_adj = None
        best_cost = float('inf')
        
        num_restarts = 8
        restart_budget = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() >= end_time:
                break
            
            # Spectral generation
            adj = spectral_generate(n, seed=42 + restart * 7)
            
            if adj is None:
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
            
            # Local search from this starting point
            adj, found = local_search(n, adj, restart_budget)
            
            final_cost = count_violations(n, adj)
            if final_cost < best_cost:
                best_cost = final_cost
                best_adj = adj.copy()
                
            if final_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges
        
        if best_adj is not None and best_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges
            
        return None

    # Try to find graphs from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Allocate time based on n
        if n <= 20:
            budget = min(remaining * 0.4, 8.0)
        elif n <= 25:
            budget = min(remaining * 0.25, 5.0)
        elif n <= 30:
            budget = min(remaining * 0.15, 3.0)
        else:
            budget = min(remaining * 0.1, 2.0)
        
        edges = try_find_graph(n, budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
