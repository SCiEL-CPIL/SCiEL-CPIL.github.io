"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Maximum Cut via multi-phase local search with tabu diversification:
    # 1. Vectorized flip-gain computation
    # 2. Greedy hill-climbing (local_improve)
    # 3. Tabu search for escaping local optima
    # 4. Simulated Annealing for further escape
    # 5. WalkSAT-style stochastic local search
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 145s wall-clock limit, return best with safety margin
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    def cut_size(labels):
        """Compute cut size: number of edges crossing the partition."""
        cross = (adj * (labels[None, :] != labels[:, None]).astype(np.int8)).sum() // 2
        return int(cross)
    
    def all_flip_gains(labels):
        """Vectorized: compute flip gain for all vertices simultaneously.
        gain[v] = (neighbors of v on same side) - (neighbors of v on opposite side)
        Flipping v changes cut by gain[v]."""
        same_side = (labels[None, :] == labels[:, None]).astype(np.int8)
        same_count = (adj * same_side).sum(axis=1)
        opp_count = (adj * (1 - same_side)).sum(axis=1)
        return same_count - opp_count
    
    def local_improve(labels, max_rounds=200):
        """Greedy hill-climbing: repeatedly flip vertices with positive gain."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline:
                break
            gains = all_flip_gains(labels)
            pos_mask = gains > 0
            if not pos_mask.any():
                break
            labels[pos_mask] = 1 - labels[pos_mask]
            improved = True
        return labels
    
    def tabu_search(labels, tabu_length=20, max_iters=20000):
        """
        Tabu search to escape local optima.
        Maintains a tabu list of recently flipped vertices to prevent cycling.
        """
        current_labels = labels.copy().astype(np.int8)
        current_cut = cut_size(current_labels)
        best_cut = current_cut
        best_labels = current_labels.copy()
        
        # Tabu list: last iteration each vertex was flipped
        tabu_last = np.full(n, -1000, dtype=np.int32)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            gains = all_flip_gains(current_labels)
            
            # Find best non-tabu move
            # Mask out tabu moves (but allow aspiration: if gain > best_cut improvement, allow)
            tabu_mask = (it - tabu_last) < tabu_length
            effective_gains = gains.copy()
            effective_gains[tabu_mask] = -np.inf
            
            best_v = int(np.argmax(effective_gains))
            best_gain = gains[best_v]
            
            # If all moves are tabu, pick the best tabu move
            if effective_gains[best_v] == -np.inf:
                best_v = int(np.argmax(gains))
                best_gain = gains[best_v]
            
            # If best gain is very negative and we're at a local optimum, 
            # occasionally do a random move to diversify
            if best_gain < -5 and rng.random() < 0.1:
                best_v = int(rng.integers(0, n))
                best_gain = gains[best_v]
            
            # Make the move
            current_labels[best_v] = 1 - current_labels[best_v]
            tabu_last[best_v] = it
            current_cut += best_gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        
        return best_labels, best_cut
    
    def simulated_annealing(labels, max_steps=30000):
        """
        Simulated Annealing for Max-Cut.
        Accept non-improving moves with probability exp(delta / T) where delta < 0.
        """
        current_labels = labels.copy().astype(np.int8)
        current_cut = cut_size(current_labels)
        best_cut = current_cut
        best_labels = current_labels.copy()
        
        T_start = max(2.0, 0.3 * n)
        T_end = 0.01
        alpha = (T_end / T_start) ** (1.0 / max(1, max_steps))
        
        T = T_start
        no_improve_count = 0
        patience = 3000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            v = int(rng.integers(0, n))
            gains = all_flip_gains(current_labels)
            delta = gains[v]
            
            if delta > 0 or (T > T_end and rng.random() < np.exp(delta / T)):
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            T *= alpha
            
            if no_improve_count > patience and T > T_end:
                T = max(T * 1.5, 1.0)
                no_improve_count = 0
        
        return best_labels, best_cut
    
    def walksat(labels, max_steps=20000, p=0.15):
        """
        WalkSAT-style stochastic local search.
        """
        current_labels = labels.copy().astype(np.int8)
        current_cut = cut_size(current_labels)
        best_cut = current_cut
        best_labels = current_labels.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            gains = all_flip_gains(current_labels)
            pos_mask = gains > 0
            
            if pos_mask.any():
                candidates = np.where(pos_mask)[0]
                if rng.random() < (1 - p):
                    v = int(candidates[rng.integers(0, len(candidates))])
                else:
                    v = int(rng.integers(0, n))
            else:
                v = int(rng.integers(0, n))
            
            current_labels[v] = 1 - current_labels[v]
            current_cut += gains[v]
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        
        return best_labels, best_cut
    
    def spectral_init():
        """Spectral partitioning using the Fiedler vector."""
        try:
            A = adj.astype(np.float64)
            D = np.diag(A.sum(axis=1))
            L = D - A
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            labels = (fiedler >= 0).astype(np.int8)
            return labels
        except Exception:
            return rng.integers(0, 2, n).astype(np.int8)
    
    def greedy_init():
        """Greedy construction: add vertices one by one, assigning to maximize local cut."""
        labels = np.zeros(n, dtype=np.int8)
        degrees = adj.sum(axis=1)
        order = np.argsort(-degrees)
        
        for idx, v in enumerate(order):
            if idx == 0:
                continue
            if time.time() > deadline:
                break
            
            side_0 = adj[v, labels == 1].sum()
            side_1 = adj[v, labels == 0].sum()
            
            if side_0 > side_1:
                labels[v] = 0
            elif side_1 > side_0:
                labels[v] = 1
            else:
                labels[v] = int(rng.integers(0, 2))
        
        return labels
    
    # Multiple restarts with diverse initializations
    best_labels = None
    best_cut = -1
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # Spectral initialization
    initial_solutions.append(spectral_init())
    
    # Greedy initialization
    initial_solutions.append(greedy_init())
    
    # Random initializations
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Degree-based initialization
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    # Alternating initialization
    initial_solutions.append((np.arange(n) % 2).astype(np.int8))
    
    # Phase 1: Local improvement from each initialization
    for init_labels in initial_solutions:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = cut_size(best_labels)
    
    # Phase 2: Tabu search from best solution
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        tabu_iters = int(min(15000, max(2000, remaining * 100)))
        tabu_labels, tabu_cut = tabu_search(best_labels, tabu_length=20, max_iters=tabu_iters)
        
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Polish after tabu
        if time.time() < deadline:
            polished = local_improve(best_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished.copy()
    
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        sa_steps = int(min(25000, max(1000, remaining * 80)))
        sa_labels, sa_cut = simulated_annealing(best_labels, max_steps=sa_steps)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish after SA
        if time.time() < deadline:
            polished = local_improve(best_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished.copy()
    
    # Phase 4: Additional restarts with perturbation + tabu
    if time.time() < deadline - 5.0 and best_labels is not None:
        remaining = deadline - time.time()
        num_extra = min(4, int(remaining // 10))
        
        for extra in range(num_extra):
            if time.time() > deadline:
                break
            
            perturbed = best_labels.copy().astype(np.int8)
            num_perturb = max(1, n // 15)
            perturbation_indices = rng.choice(n, size=num_perturb, replace=False)
            perturbed[perturbation_indices] = 1 - perturbed[perturbation_indices]
            
            # Local improve
            polished = local_improve(perturbed)
            pol_cut = cut_size(polished)
            
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished.copy()
            
            # Tabu from perturbed
            if time.time() < deadline - 3.0:
                tabu_remaining = deadline - time.time()
                tabu_iters = int(min(8000, max(1000, tabu_remaining * 80)))
                tabu_labels, tabu_cut = tabu_search(polished, tabu_length=15, max_iters=tabu_iters)
                
                if tabu_cut > best_cut:
                    best_cut = tabu_cut
                    best_labels = tabu_labels.copy()
                
                if time.time() < deadline:
                    final_polish = local_improve(best_labels)
                    fp_cut = cut_size(final_polish)
                    if fp_cut > best_cut:
                        best_cut = fp_cut
                        best_labels = final_polish.copy()
    
    # Final polish
    if time.time() < deadline - 1.0 and best_labels is not None:
        final_labels = local_improve(best_labels)
        final_cut = cut_size(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
