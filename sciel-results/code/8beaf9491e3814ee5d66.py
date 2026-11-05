"""Rewritten program for Maximum Cut using a direct structural approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a combination of:
      1. Spectral initialization (Laplacian eigenvector)
      2. Fast local search (greedy hill-climbing)
      3. Simulated annealing for escape from local optima
      4. Multi-start with random restarts
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

        # Cut size: number of edges with endpoints in different partitions
        def cut_size(labels):
            """Fast cut size: count edges crossing the cut."""
            diff = labels[:, None] != labels[None, :]
            return int(np.sum(adj * diff)) // 2

        def local_search(labels, max_rounds=500):
            """Greedy hill-climbing: flip any vertex that improves cut.
            Uses sequential updates with accurate gain calculation."""
            best_labels = labels.copy()
            best_cut = cut_size(best_labels)
            
            for rnd in range(max_rounds):
                if time_left() < 2.0:
                    break
                improved = False
                # Try flipping each vertex once in random order
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    
                    # Calculate gain of flipping v accurately
                    # same: neighbors with same label
                    # opp: neighbors with different label
                    # delta = same - opp
                    mask = (best_labels == best_labels[v])
                    same = np.sum(adj[v, mask])
                    opp = np.sum(adj[v, ~mask])
                    delta = int(same - opp)
                    
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                
                if not improved:
                    break
            return best_labels, best_cut

        def simulated_annealing(labels, max_iters=50000, T_start=4.0, T_end=0.05, cooling=0.99997):
            """Simulated annealing to escape local optima."""
            best_labels = labels.copy()
            best_cut = cut_size(best_labels)
            current_labels = labels.copy()
            current_cut = best_cut
            T = T_start

            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                v = np.random.randint(0, n)
                
                # Calculate gain of flipping v accurately
                mask = (current_labels == current_labels[v])
                same = np.sum(adj[v, mask])
                opp = np.sum(adj[v, ~mask])
                delta = int(same - opp)

                if delta >= 0:
                    # Accept
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
                else:
                    # Accept with probability exp(delta / T)
                    if T > 0.001:
                        prob = np.exp(delta / T)
                        if np.random.random() < prob:
                            current_labels[v] = 1 - current_labels[v]
                            current_cut += delta

                T = max(T * cooling, T_end)

            return best_labels, best_cut

        def quantum_annealing(labels, max_iters=80000, gamma_start=10.0, gamma_end=0.1):
            """Quantum-inspired annealing using transverse-field tunneling.
            
            The transverse field s(t) modulates the probability of accepting
            worse moves. Unlike classical SA which uses temperature to accept
            worse states, quantum annealing uses a transverse field that allows
            tunneling through energy barriers. We approximate this by:
            1. Using a different acceptance criterion based on the transverse field
            2. Allowing correlated multi-flip moves when the field is strong
            3. The acceptance probability increases with the transverse field
               for moves that would be rejected by classical SA
            """
            best_labels = labels.copy()
            best_cut = cut_size(best_labels)
            current_labels = labels.copy()
            current_cut = best_cut
            
            # Precompute degree for faster gain calculation
            deg = degrees.copy()
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Transverse field schedule: linear interpolation from gamma_start to gamma_end
                progress = it / max(1, max_iters)
                s = gamma_start * (1 - progress) + gamma_end * progress
                
                # With high transverse field, allow larger perturbations (multi-flips)
                # This approximates quantum tunneling through barriers
                if s > 5.0:
                    # High tunneling: flip 1-3 vertices
                    n_flips = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                elif s > 2.0:
                    # Medium tunneling: flip 1-2 vertices
                    n_flips = np.random.choice([1, 2], p=[0.7, 0.3])
                else:
                    # Low tunneling: mostly single flips
                    n_flips = 1
                
                # Select vertices to flip
                if n_flips == 1:
                    v = np.random.randint(0, n)
                    mask = (current_labels == current_labels[v])
                    same = np.sum(adj[v, mask])
                    opp = np.sum(adj[v, ~mask])
                    delta = int(same - opp)
                    
                    if delta >= 0:
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        # Quantum acceptance: probability depends on transverse field
                        # Higher s means more tunneling, so accept worse moves more often
                        # Use a modified Boltzmann-like acceptance with quantum factor
                        if s > 0.1:
                            # Quantum tunneling probability: exp(-|delta| / (s * 2))
                            # This is larger than classical SA for the same "temperature"
                            prob = np.exp(-abs(delta) / (s * 2.0))
                            if np.random.random() < prob:
                                current_labels[v] = 1 - current_labels[v]
                                current_cut += delta
                else:
                    # Multi-flip move (quantum tunneling approximation)
                    vertices = np.random.choice(n, size=n_flips, replace=False)
                    # Calculate total delta for the multi-flip
                    old_labels = current_labels[vertices].copy()
                    total_delta = 0
                    for v in vertices:
                        mask = (current_labels == current_labels[v])
                        same = np.sum(adj[v, mask])
                        opp = np.sum(adj[v, ~mask])
                        total_delta += int(same - opp)
                    
                    if total_delta >= 0:
                        current_labels[vertices] = 1 - current_labels[vertices]
                        current_cut += total_delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        # Quantum acceptance for multi-flip
                        if s > 0.1:
                            prob = np.exp(-abs(total_delta) / (s * 3.0))
                            if np.random.random() < prob:
                                current_labels[vertices] = 1 - current_labels[vertices]
                                current_cut += total_delta

            return best_labels, best_cut

        def spectral_init():
            """Use the Laplacian eigenvector to initialize the partition."""
            try:
                D = np.diag(degrees)
                L = D - adj.astype(np.float64)
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, 1])
                v = evecs[:, 1]
                labels = (v > 0).astype(int)
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)
                return labels
            except Exception:
                return np.random.randint(0, 2, size=n)

        # Strategy:
        # 1. Spectral initialization + local search
        # 2. Multiple random restarts with local search
        # 3. Simulated annealing from best solutions
        # 4. Final polish

        best_labels = None
        best_cut = -1

        # Phase 1: Spectral init + local search
        labels_spec = spectral_init()
        labels_ls, cut_ls = local_search(labels_spec, max_rounds=200)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

        # Phase 2: Random restarts with local search
        num_restarts = 30
        for r in range(num_restarts):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            labels_rs, cut_rs = local_search(labels_rand, max_rounds=100)
            if cut_rs > best_cut:
                best_cut = cut_rs
                best_labels = labels_rs.copy()

        # Phase 3: Simulated annealing from best
        if time_left() > 10.0 and best_labels is not None:
            labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=80000)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()

        # Phase 3.5: Quantum annealing from best
        if time_left() > 15.0 and best_labels is not None:
            labels_qa, cut_qa = quantum_annealing(best_labels, max_iters=80000, gamma_start=10.0, gamma_end=0.1)
            if cut_qa > best_cut:
                best_cut = cut_qa
                best_labels = labels_qa.copy()

        # Phase 4: More random restarts with SA and QA if time permits
        if time_left() > 20.0:
            for r in range(8):
                if time_left() < 5.0:
                    break
                labels_rand = np.random.randint(0, 2, size=n)
                # Alternate between SA and QA for diversity
                if r % 2 == 0:
                    labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=30000)
                else:
                    labels_sa2, cut_sa2 = quantum_annealing(labels_rand, max_iters=30000, gamma_start=8.0, gamma_end=0.1)
                if cut_sa2 > best_cut:
                    best_cut = cut_sa2
                    best_labels = labels_sa2.copy()

        # Phase 5: Final polish with local search
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(15):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break
        
        # Phase 6: Final quantum polish if time permits
        if time_left() > 5.0 and best_labels is not None:
            labels_qp, cut_qp = quantum_annealing(best_labels, max_iters=30000, gamma_start=5.0, gamma_end=0.05)
            if cut_qp > best_cut:
                best_cut = cut_qp
                best_labels = labels_qp.copy()
            # Final local search after quantum polish
            if time_left() > 2.0:
                labels_polish2, cut_polish2 = local_search(best_labels, max_rounds=50)
                if cut_polish2 > best_cut:
                    best_cut = cut_polish2
                    best_labels = labels_polish2.copy()

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
