"""Rewritten program for Maximum Cut using Parallel Tempering (Replica Exchange)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a Parallel Tempering (Replica Exchange) approach with:
      1. Spectral initialization
      2. Vectorized gain computation for single-flip moves
      3. Population of replicas at different temperatures
      4. Periodic state exchanges between replicas
      5. Final high-precision polish pass
    Deadline guard ensures return well before 150s.
    """
    try:
        adj = np.asarray(adjacency, dtype=np.int32)
        n = adj.shape[0]
        if n == 0:
            return []

        # Ensure symmetric and no self loops for safety
        adj = (adj + adj.T) // 2
        np.fill_diagonal(adj, 0)

        start_time = time.time()
        DEADLINE = start_time + 140.0  # Safety margin before 150s kill

        def time_left():
            return DEADLINE - time.time()

        # Precompute
        degrees = adj.sum(axis=1)
        E = int(degrees.sum()) // 2
        adj_float = adj.astype(np.float64)
        degrees_float = degrees.astype(np.float64)

        def cut_size(labels):
            """Fast cut size: count edges crossing the cut."""
            diff = labels[:, None] != labels[None, :]
            return int(np.sum(adj * diff)) // 2

        def compute_gains(labels):
            """Vectorized computation of gain for flipping each vertex.
            gain[v] = 2 * (sum of adj[v,u] where labels[u] == labels[v]) - degrees[v]
            If gain > 0, flipping v increases the cut."""
            l0_mask = (labels == 0).astype(np.int8)
            l1_mask = (labels == 1).astype(np.int8)
            
            adj_l0 = adj_float @ l0_mask
            adj_l1 = adj_float @ l1_mask
            
            same_count = np.where(labels == 0, adj_l0, adj_l1)
            gains = 2.0 * same_count - degrees_float
            return gains

        def local_search(labels, max_rounds=200):
            """Greedy hill-climbing with vectorized gain computation."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            for rnd in range(max_rounds):
                if time_left() < 2.0:
                    break
                improved = False
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    
                    gains = compute_gains(best_labels)
                    delta = int(gains[v])
                    
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                
                if not improved:
                    break
            return best_labels, best_cut

        def spectral_init():
            """Use the Laplacian eigenvector to initialize the partition."""
            try:
                D = np.diag(degrees)
                L = D - adj_float
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, 1])
                v = evecs[:, 1]
                labels = (v > 0).astype(int)
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)
                return labels
            except Exception:
                return np.random.randint(0, 2, size=n)

        def parallel_tempering(initial_labels, num_replicas=8, max_iters=100000, 
                               T_start=4.0, T_end=0.05, exchange_interval=100):
            """
            Parallel Tempering (Replica Exchange) algorithm.
            Maintains a population of replicas at different temperatures.
            Periodically exchanges states between replicas to escape local optima.
            """
            # Initialize replicas
            replica_labels = []
            replica_cuts = []
            replica_temps = []
            
            # Geometric temperature schedule
            T_values = np.geomspace(T_start, T_end, num_replicas)
            
            for i in range(num_replicas):
                if i == 0:
                    # First replica starts from the best initial solution
                    lab = initial_labels.copy().astype(np.int8)
                else:
                    # Other replicas start from perturbed versions
                    lab = initial_labels.copy().astype(np.int8)
                    # Randomly flip some vertices
                    num_flips = int(n * 0.3)
                    flip_indices = np.random.choice(n, size=num_flips, replace=False)
                    lab[flip_indices] = 1 - lab[flip_indices]
                
                replica_labels.append(lab)
                replica_cuts.append(cut_size(lab))
                replica_temps.append(T_values[i])
            
            best_labels = initial_labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # For each replica, perform a single-flip move
                for r in range(num_replicas):
                    if time_left() < 1.0:
                        break
                    
                    v = np.random.randint(0, n)
                    gains = compute_gains(replica_labels[r])
                    delta = int(gains[v])
                    T = replica_temps[r]
                    
                    if delta >= 0:
                        # Accept
                        replica_labels[r][v] = 1 - replica_labels[r][v]
                        replica_cuts[r] += delta
                        if replica_cuts[r] > best_cut:
                            best_cut = replica_cuts[r]
                            best_labels = replica_labels[r].copy()
                    else:
                        # Accept with probability exp(delta / T)
                        if T > 0.001:
                            prob = np.exp(delta / T)
                            if np.random.random() < prob:
                                replica_labels[r][v] = 1 - replica_labels[r][v]
                                replica_cuts[r] += delta
                
                # Periodically exchange states between replicas
                if it % exchange_interval == 0 and it > 0:
                    # Try to exchange between adjacent replicas
                    for r in range(num_replicas - 1):
                        if time_left() < 1.0:
                            break
                        
                        T_high = replica_temps[r]      # higher temperature (lower index)
                        T_low = replica_temps[r + 1]   # lower temperature (higher index)
                        
                        # Metropolis criterion for exchange
                        # delta_E = E_low - E_high (cut sizes, so higher is better)
                        # We want to accept if the higher-energy (lower-cut) replica at low T
                        # swaps with the lower-energy (higher-cut) replica at high T
                        delta_cut = replica_cuts[r] - replica_cuts[r + 1]
                        
                        # Accept with probability min(1, exp(-delta_cut / T_low))
                        # Since we're maximizing cut, we accept if delta_cut < 0 (swap improves)
                        if delta_cut <= 0:
                            # Swap is beneficial, always accept
                            replica_labels[r], replica_labels[r + 1] = replica_labels[r + 1].copy(), replica_labels[r].copy()
                            replica_cuts[r], replica_cuts[r + 1] = replica_cuts[r + 1], replica_cuts[r]
                        else:
                            # Swap is detrimental, accept with probability exp(-delta_cut / T_low)
                            if T_low > 0.001:
                                prob = np.exp(-delta_cut / T_low)
                                if np.random.random() < prob:
                                    replica_labels[r], replica_labels[r + 1] = replica_labels[r + 1].copy(), replica_labels[r].copy()
                                    replica_cuts[r], replica_cuts[r + 1] = replica_cuts[r + 1], replica_cuts[r]
            
            return best_labels, best_cut

        def goptimal_polish(labels):
            """
            A more aggressive local search that tries all single flips,
            then all pairs of flips, to find a better local optimum.
            """
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            # First, do standard hill climbing
            for rnd in range(200):
                if time_left() < 2.0:
                    break
                improved = False
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    gains = compute_gains(best_labels)
                    delta = int(gains[v])
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                if not improved:
                    break
            
            # Try pairwise flips
            for rnd in range(10):
                if time_left() < 2.0:
                    break
                improved = False
                pairs = np.random.randint(0, n, size=(min(200, n * n // 2), 2))
                for i, j in pairs:
                    if i == j:
                        continue
                    if time_left() < 1.0:
                        break
                    
                    old_i = best_labels[i]
                    old_j = best_labels[j]
                    best_labels[i] = 1 - old_i
                    best_labels[j] = 1 - old_j
                    new_cut = cut_size(best_labels)
                    delta = new_cut - best_cut
                    if delta > 0:
                        best_cut = new_cut
                        improved = True
                    else:
                        best_labels[i] = old_i
                        best_labels[j] = old_j
                
                if not improved:
                    break
            
            return best_labels, best_cut

        # Strategy:
        # 1. Spectral initialization + local search
        # 2. Parallel Tempering from best solution
        # 3. Multiple random restarts with Parallel Tempering
        # 4. Final high-precision polish

        best_labels = None
        best_cut = -1

        # Phase 1: Spectral init + local search
        labels_spec = spectral_init()
        labels_ls, cut_ls = local_search(labels_spec, max_rounds=200)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

        # Phase 2: Parallel Tempering from best solution
        if time_left() > 20.0 and best_labels is not None:
            labels_pt, cut_pt = parallel_tempering(best_labels, num_replicas=8, max_iters=80000)
            if cut_pt > best_cut:
                best_cut = cut_pt
                best_labels = labels_pt.copy()
            # Polish after Parallel Tempering
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 3: Random restarts with Parallel Tempering
        num_restarts = 10
        for r in range(num_restarts):
            if time_left() < 10.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            labels_pt_r, cut_pt_r = parallel_tempering(labels_rand, num_replicas=6, max_iters=40000)
            if cut_pt_r > best_cut:
                best_cut = cut_pt_r
                best_labels = labels_pt_r.copy()
            # Polish
            if time_left() > 3.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4: More random restarts with local search if time permits
        if time_left() > 15.0:
            for r in range(15):
                if time_left() < 3.0:
                    break
                labels_rand = np.random.randint(0, 2, size=n)
                labels_ls_r, cut_ls_r = local_search(labels_rand, max_rounds=100)
                if cut_ls_r > best_cut:
                    best_cut = cut_ls_r
                    best_labels = labels_ls_r.copy()

        # Phase 5: Final high-precision polish
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(20):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = goptimal_polish(best_labels)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break

        if best_labels is None:
            best_labels = np.random.randint(0, 2, size=n)

        return best_labels.tolist()

    except Exception as e:
        # Fallback to random partition if anything fails
        try:
            n = np.asarray(adjacency).shape[0]
            return np.random.randint(0, 2, size=n).tolist()
        except:
            return []
# EVOLVE-BLOCK-END
