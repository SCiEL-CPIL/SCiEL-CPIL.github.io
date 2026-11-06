"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    total_edges = adj.sum() // 2
    
    def cut_size(labels):
        """Compute cut size from binary labels."""
        s = 2 * labels - 1
        val = s @ (adj @ s)
        return total_edges - val // 2
    
    def compute_gains(labels):
        """Vectorized computation of flip gains for all vertices."""
        s = 2 * labels - 1
        # gain[v] = 2 * (edges from v to same side) - degree[v]
        # edges from v to same side = sum(adj[v,u] * (s[u]==s[v]))
        same_mask = (s[:, None] == s[None, :]).astype(np.int8)
        same_counts = (adj * same_mask).sum(axis=1)
        degs = adj.sum(axis=1)
        gains = 2 * same_counts - degs
        return gains
    
    def local_search(labels, time_limit=None, max_iters=50000):
        """Fast local search with vectorized gain updates."""
        labels = labels.copy()
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            gains = compute_gains(labels)
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            if best_gain <= 0:
                break
            labels[best_v] = 1 - labels[best_v]
        return labels
    
    def spectral_init(adj, n):
        """Spectral initialization using Fiedler vector (second eigenvector of Laplacian)."""
        # Build Laplacian L = D - A
        deg = adj.sum(axis=1).astype(float)
        L = np.diag(deg) - adj.astype(float)
        
        try:
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # eigh returns in ascending order, so index 1
            fiedler = eigenvectors[:, 1]
            # Map to 0/1 labels: positive -> 1, negative -> 0
            labels = (fiedler >= 0).astype(int)
            return labels
        except:
            # Fallback to random if spectral fails
            return np.random.randint(0, 2, size=n)
    
    def greedy_init(adj, n):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        np.random.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels
    
    def sa_from_start(labels, adj, n, t_start, t_end, iters, time_limit=None):
        """Simulated annealing."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(iters, 1))
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            
            v = random.randint(0, n - 1)
            s = 2 * labels - 1
            same_side = (adj[v] * (s == s[v])).sum()
            deg = adj[v].sum()
            gain = 2 * same_side - deg
            
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    def pair_flip_search(labels, time_limit=None, max_iters=20000):
        """Try flipping pairs of vertices to escape local optima."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            # Pick two random vertices
            v1 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if v1 == v2:
                continue
            
            # Compute gain for flipping both v1 and v2
            s = 2 * labels - 1
            # Gain for flipping v1 alone
            same1 = (adj[v1] * (s == s[v1])).sum()
            deg1 = adj[v1].sum()
            gain1 = 2 * same1 - deg1
            
            # Gain for flipping v2 alone
            same2 = (adj[v2] * (s == s[v2])).sum()
            deg2 = adj[v2].sum()
            gain2 = 2 * same2 - deg2
            
            # If v1 and v2 are connected, flipping both changes the edge between them
            # The combined gain is more complex; let's just compute it directly
            new_labels = labels.copy()
            new_labels[v1] = 1 - new_labels[v1]
            new_labels[v2] = 1 - new_labels[v2]
            new_cut = cut_size(new_labels)
            
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = new_labels.copy()
                labels = new_labels
        
        return best_labels, best_cut
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse starts with local search
    starts = []
    
    # Start 1: Spectral initialization (Fiedler vector)
    starts.append(spectral_init(adj, n))
    
    # Start 2: Alternating
    starts.append(np.arange(n) % 2)
    
    # Start 3: Greedy
    starts.append(greedy_init(adj, n))
    
    # Starts 4-12: Random
    for _ in range(9):
        starts.append(np.random.randint(0, 2, size=n))
    
    for i, init_labels in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Local search from this start
        ls_labels = local_search(init_labels, time_limit=deadline)
        c = cut_size(ls_labels)
        
        if c > best_cut:
            best_cut = c
            best_labels = ls_labels.copy()
    
    # Phase 2: Simulated annealing from best solution
    if time.time() < deadline - 20:
        sa_time_limit = deadline - 20
        sa_labels, sa_cut = sa_from_start(
            best_labels, adj, n, 
            t_start=3.0, t_end=0.01, 
            iters=300000, 
            time_limit=sa_time_limit
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: Pair flip search to escape local optima
    if time.time() < deadline - 15:
        pf_labels, pf_cut = pair_flip_search(
            best_labels, 
            time_limit=deadline - 15,
            max_iters=50000
        )
        if pf_cut > best_cut:
            best_cut = pf_cut
            best_labels = pf_labels.copy()
        
        # Local search after pair flips
        if time.time() < deadline - 10:
            ls2 = local_search(best_labels, time_limit=deadline - 10)
            c2 = cut_size(ls2)
            if c2 > best_cut:
                best_cut = c2
                best_labels = ls2.copy()
    
    # Phase 4: Final local search polish
    if time.time() < deadline:
        final_labels = local_search(best_labels, time_limit=deadline)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    # Phase 5: One more SA pass if time allows
    if time.time() < deadline - 10:
        sa2_labels, sa2_cut = sa_from_start(
            best_labels, adj, n,
            t_start=1.5, t_end=0.005,
            iters=200000,
            time_limit=deadline - 10
        )
        if sa2_cut > best_cut:
            best_cut = sa2_cut
            best_labels = sa2_labels.copy()
        
        # Final polish
        if time.time() < deadline:
            final2 = local_search(best_labels, time_limit=deadline)
            final2_cut = cut_size(final2)
            if final2_cut > best_cut:
                best_labels = final2
    
    return best_labels
# EVOLVE-BLOCK-END
