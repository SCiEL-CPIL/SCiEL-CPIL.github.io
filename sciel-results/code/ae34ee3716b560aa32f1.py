"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Maximum Cut via multi-phase local search:
    # 1. Vectorized flip-gain computation
    # 2. Greedy hill-climbing (local_improve)
    # 3. WalkSAT-style stochastic local search (Max-2-SAT formulation)
    # 4. Simulated Annealing for escape from local optima
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 138.0
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
        # For vertex v, same_side[v, u] = 1 if u is on same side as v
        # adj[v, u] * same_side[v, u] = 1 if u is neighbor AND same side
        same_count = (adj * same_side).sum(axis=1)  # neighbors on same side for each v
        opp_count = (adj * (1 - same_side)).sum(axis=1)  # neighbors on opposite side
        # When we flip v: same-side neighbors become opposite (contribute to cut),
        # opposite-side neighbors become same (no longer contribute)
        # New cut contribution from v's edges = same_count[v]
        # Old cut contribution from v's edges = opp_count[v]
        # Delta = same_count[v] - opp_count[v]
        return same_count - opp_count
    
    def local_improve(labels):
        """Greedy hill-climbing: repeatedly flip vertices with positive gain."""
        improved = True
        max_rounds = 200
        while improved and max_rounds > 0:
            improved = False
            max_rounds -= 1
            gains = all_flip_gains(labels)
            pos_mask = gains > 0
            if not pos_mask.any():
                break
            # Flip all vertices with positive gain simultaneously
            labels[pos_mask] = 1 - labels[pos_mask]
            improved = True
        return labels
    
    def walksat(labels, max_steps=30000, p=0.1):
        """
        WalkSAT-style stochastic local search for Max-2-SAT formulation.
        
        Max-Cut as Max-2-SAT: For each edge (u,v), clause (x_u != x_v).
        Equivalent to: for edge (u,v), we want x_u XOR x_v = 1.
        Unsatisfied clause = edge with both endpoints on same side (uncut edge).
        
        Strategy:
        - If there's a variable that, when flipped, satisfies more clauses than it breaks,
          flip it with probability (1-p).
        - Otherwise, pick a random variable from those that break the most clauses
          (i.e., are in unsatisfied clauses) and flip it with probability p.
        
        This creates a random walk that can escape local optima.
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
                # Choose a random vertex with positive gain
                candidates = np.where(pos_mask)[0]
                v = candidates[rng.integers(0, len(candidates))]
                # With probability (1-p), flip a good vertex; with prob p, do random walk
                if rng.random() < (1 - p):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gains[v]
                else:
                    # Random walk: pick a random vertex
                    v = rng.integers(0, n)
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gains[v]
            else:
                # At local optimum: all flips are non-improving
                # Pick a random vertex to flip (random walk step)
                v = rng.integers(0, n)
                current_labels[v] = 1 - current_labels[v]
                current_cut += gains[v]
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        
        return best_labels, best_cut
    
    def simulated_annealing(labels, max_steps=50000):
        """
        QUBO Simulated Annealing.
        Energy E = -cut_size. We minimize E (maximize cut).
        Accept non-improving moves with probability exp(-delta_E / T).
        """
        current_labels = labels.copy().astype(np.int8)
        current_cut = cut_size(current_labels)
        best_cut = current_cut
        best_labels = current_labels.copy()
        
        # Temperature schedule: geometric cooling
        T_start = max(1.0, 0.5 * n)  # initial temperature
        T_end = 0.01
        alpha = (T_end / T_start) ** (1.0 / max_steps) if max_steps > 1 else 1.0
        
        T = T_start
        no_improve_count = 0
        patience = 5000  # restart cooling if stuck
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = rng.integers(0, n)
            
            # Compute delta in cut size
            gains = all_flip_gains(current_labels)
            delta = gains[v]
            
            # delta > 0: improvement (lower energy)
            # delta <= 0: worsening (higher energy)
            if delta > 0 or (T > T_end and rng.random() < np.exp(-(-delta) / T)):
                # Accept move
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Cool down
            T *= alpha
            
            # If stuck for too long, reheat slightly to escape local optimum
            if no_improve_count > patience and T > T_end:
                T = max(T * 2.0, 1.0)
                no_improve_count = 0
        
        return best_labels, best_cut
    
    def spectral_init():
        """Spectral partitioning using the Fiedler vector."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(adj.astype(np.float64))
            fiedler = eigenvectors[:, 1]
            labels = (fiedler > 0).astype(np.int8)
            return labels
        except Exception:
            return rng.integers(0, 2, n).astype(np.int8)
    
    def greedy_init():
        """Greedy construction: add vertices one by one, assigning to maximize local cut."""
        labels = np.zeros(n, dtype=np.int8)
        order = rng.permutation(n)
        
        for idx, v in enumerate(order):
            if idx == 0:
                continue
            
            side_0 = 0
            side_1 = 0
            for u in order[:idx]:
                if adj[v, u]:
                    if labels[u] == 0:
                        side_0 += 1
                    else:
                        side_1 += 1
            
            if side_0 < side_1:
                labels[v] = 1
            elif side_1 < side_0:
                labels[v] = 0
            else:
                labels[v] = rng.integers(0, 2)
        
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
    
    # Track time per restart to ensure we fit within deadline
    start_time = time.time()
    total_restarts = len(initial_solutions)
    time_per_restart = (deadline - start_time) / total_restarts
    
    for init_idx, init_labels in enumerate(initial_solutions):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        
        # Phase 1: Greedy local improvement
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # Phase 2: WalkSAT-style stochastic local search
        remaining_time = deadline - time.time()
        ws_steps = int(min(30000, max(2000, remaining_time * 600)))
        
        ws_labels, ws_cut = walksat(labels, max_steps=ws_steps, p=0.15)
        
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels.copy()
        
        # Polish after WalkSAT
        if time.time() < deadline:
            polished = local_improve(ws_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished.copy()
        
        # Phase 3: Simulated Annealing
        if time.time() < deadline:
            remaining_time = deadline - time.time()
            sa_steps = int(min(40000, max(1000, remaining_time * 500)))
            
            sa_labels, sa_cut = simulated_annealing(best_labels, max_steps=sa_steps)
            
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
            
            # Final polish
            if time.time() < deadline:
                polished = local_improve(sa_labels)
                pol_cut = cut_size(polished)
                if pol_cut > best_cut:
                    best_cut = pol_cut
                    best_labels = polished.copy()
    
    # If we have time left, do additional WalkSAT + SA restarts from the best solution
    if time.time() < deadline and best_labels is not None:
        remaining = deadline - time.time()
        if remaining > 5.0:
            extra_steps = int(min(25000, max(2000, remaining * 400)))
            for extra in range(4):
                if time.time() > deadline:
                    break
                perturbed = best_labels.copy().astype(np.int8)
                # Perturb a few random vertices
                num_perturb = max(1, n // 10)
                perturbation_indices = rng.choice(n, size=num_perturb, replace=False)
                perturbed[perturbation_indices] = 1 - perturbed[perturbation_indices]
                
                # WalkSAT phase
                ws_labels, ws_cut = walksat(perturbed, max_steps=extra_steps, p=0.12)
                if ws_cut > best_cut:
                    best_cut = ws_cut
                    best_labels = ws_labels.copy()
                
                # Polish
                if time.time() < deadline:
                    polished = local_improve(ws_labels)
                    pol_cut = cut_size(polished)
                    if pol_cut > best_cut:
                        best_cut = pol_cut
                        best_labels = polished.copy()
                
                # SA phase
                if time.time() < deadline:
                    sa_remaining = deadline - time.time()
                    if sa_remaining > 3.0:
                        sa_steps = int(min(20000, max(1000, sa_remaining * 300)))
                        sa_labels, sa_cut = simulated_annealing(best_labels, max_steps=sa_steps)
                        if sa_cut > best_cut:
                            best_cut = sa_cut
                            best_labels = sa_labels.copy()
                        
                        if time.time() < deadline:
                            polished = local_improve(sa_labels)
                            pol_cut = cut_size(polished)
                            if pol_cut > best_cut:
                                best_cut = pol_cut
                                best_labels = polished.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
