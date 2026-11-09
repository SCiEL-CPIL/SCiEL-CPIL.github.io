import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s if evaluator is stricter)
    
    # --- Data Structures ---
    # Use bitmasks for adjacency for fast intersection checks
    adj = [0] * n
    degs = [0] * n
    
    def get_edges_from_adj():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def calc_true_score():
        """Calculate the maximum triangle degree sum."""
        max_sum = 0
        for u in range(n):
            if adj[u] == 0:
                continue
            mask_v = adj[u]
            # Only consider v > u
            mask_v &= ~((1 << (u + 1)) - 1) # Clear bits 0..u
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                # Iterate over common neighbors w
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb_w
        return max_sum

    def calc_penalty():
        """Calculate the sum of squares of triangle degree sums."""
        penalty = 0
        for u in range(n):
            if adj[u] == 0:
                continue
            mask_v = adj[u]
            # Only consider v > u
            mask_v &= ~((1 << (u + 1)) - 1) # Clear bits 0..u
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                # Iterate over common neighbors w
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    penalty += s * s
                    common ^= lsb_w
        return penalty

    def make_swap(u1, v1, u2, v2):
        """Remove edge (u1,v1), add edge (u2,v2). Updates adj and degs in place."""
        # Remove
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        
        # Add
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1

    def unmake_swap(u1, v1, u2, v2):
        """Revert the swap."""
        # Remove added
        adj[u2] &= ~(1 << v2)
        adj[v2] &= ~(1 << u2)
        degs[u2] -= 1
        degs[v2] -= 1
        
        # Add removed
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        degs[u1] += 1
        degs[v1] += 1

    def spectral_init():
        """
        Initialize a graph using a spectral heuristic.
        1. Generate a random graph with target_edges.
        2. Compute Laplacian and Fiedler vector.
        3. Score all possible edges by |v_u - v_v|.
        4. Select top 'target_edges' edges.
        """
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
            
        # 1. Random graph for Laplacian
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        A = np.zeros((n, n))
        for u, v in all_pairs[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
            
        # 2. Laplacian
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # 3. Eigen decomposition
        # We want the Fiedler vector (eigenvector of 2nd smallest eigenvalue)
        # For connected graphs, lambda_0 = 0, lambda_1 > 0.
        # If graph is disconnected, lambda_1 = 0. We should handle this.
        # But random graph with 65 edges on 16 vertices is almost surely connected.
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find index of smallest positive eigenvalue
        # eigvals are sorted ascending.
        # If eigvals[0] is ~0, we want eigvals[1].
        # If graph is disconnected, eigvals[1] might also be ~0.
        # In that case, we pick the first eigenvector with significant variance.
        
        fiedler_vec = None
        for i in range(1, n):
            if eigvals[i] > 1e-6:
                fiedler_vec = eigvecs[:, i]
                break
        
        if fiedler_vec is None:
            # Fallback to random init if spectral fails
            for u, v in all_pairs[:target_edges]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
            return

        # 4. Score edges
        edge_scores = []
        for u, v in all_pairs:
            score = abs(fiedler_vec[u] - fiedler_vec[v])
            edge_scores.append((score, u, v))
            
        # Sort by score descending
        edge_scores.sort(reverse=True)
        
        # Select top target_edges
        for score, u, v in edge_scores[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1

    def random_graph_init():
        """Initialize a random graph with exactly target_edges edges."""
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1

    # --- Search Strategy ---
    
    best_score = float('inf')
    best_adj = [0] * n
    best_degs = [0] * n
    best_edges = None
    
    # Pre-compute list of all possible edges
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    num_restarts = 50 # Increased restarts, spectral init is fast
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization: Alternate between Spectral and Random
        if r % 2 == 0:
            spectral_init()
        else:
            random_graph_init()
            
        current_penalty = calc_penalty()
        current_true_score = calc_true_score()
        
        if current_true_score < best_score:
            best_score = current_true_score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj()
            
        # 2. Local Search (Simulated Annealing)
        # Optimize the penalty (sum of squares) to flatten distribution
        T = 5.0 # Higher temp for penalty which has larger scale
        cooling = 0.99
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick edge (u1, v1) to remove
            while True:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break
            
            # Pick non-edge (u2, v2) to add
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    if u2 == u1 and v2 == v1:
                        continue
                    break
            
            # Apply swap
            make_swap(u1, v1, u2, v2)
            new_penalty = calc_penalty()
            
            delta = new_penalty - current_penalty
            
            if delta <= 0:
                # Improvement in penalty
                current_penalty = new_penalty
                no_improve = 0
                
                # Check true score for best record
                new_true_score = calc_true_score()
                if new_true_score < best_score:
                    best_score = new_true_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj()
            else:
                # Worse penalty
                if T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_penalty = new_penalty
                        no_improve = 0
                        # Note: We don't check true score here to save time, 
                        # unless penalty improvement is likely to correlate.
                        # But if we accept a worse penalty, true score might also be worse.
                        # However, occasionally it might be better. 
                        # Let's check true score only if penalty didn't improve significantly?
                        # To be safe and find the best true score, let's check.
                        new_true_score = calc_true_score()
                        if new_true_score < best_score:
                            best_score = new_true_score
                            best_adj = adj[:]
                            best_degs = degs[:]
                            best_edges = get_edges_from_adj()
                    else:
                        # Revert
                        unmake_swap(u1, v1, u2, v2)
                        no_improve += 1
                else:
                    # Revert
                    unmake_swap(u1, v1, u2, v2)
                    no_improve += 1
            
            T *= cooling
            
        # End of local search for this restart
        
    # Fallback if no valid graph found (should not happen)
    if best_edges is None:
        # Construct seed graph
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}
