"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # QUBO-based Simulated Annealing for Max-Cut
    # Energy = -cut_size (minimize energy = maximize cut)
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 138.0
    rng = np.random.default_rng(42)
    
    def cut_size(labels):
        """Compute cut size: number of edges crossing the partition."""
        # For each vertex, count neighbors with different label
        # Sum and divide by 2 (each edge counted twice)
        cross = np.zeros(n, dtype=np.int32)
        for v in range(n):
            cross[v] = adj[v][labels != labels[v]].sum()
        return int(cross.sum() // 2)
    
    def flip_gain(v, labels):
        """Compute change in cut size if we flip vertex v."""
        # Current: edges from v to opposite side contribute
        # After flip: edges from v to same side (now opposite) contribute
        cur_cross = adj[v][labels != labels[v]].sum()
        new_cross = adj[v][labels == labels[v]].sum()
        return int(new_cross - cur_cross)
    
    def local_improve(labels):
        """Greedy hill-climbing: flip any vertex that improves cut."""
        improved = True
        while improved:
            improved = False
            for v in range(n):
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
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
            delta = flip_gain(v, current_labels)
            
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
        # Compute eigenvalues/vectors of the adjacency matrix
        # Use the second eigenvector (Fiedler vector) for partitioning
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(adj.astype(np.float64))
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # eigh returns in ascending order, so index 1 is the Fiedler vector
            fiedler = eigenvectors[:, 1]
            # Partition: vertices with positive fiedler value -> 1, negative -> 0
            labels = (fiedler > 0).astype(np.int8)
            return labels
        except Exception:
            return rng.integers(0, 2, n).astype(np.int8)
    
    def greedy_init():
        """Greedy construction: add vertices one by one, assigning to maximize local cut."""
        labels = np.zeros(n, dtype=np.int8)
        order = rng.permutation(n)
        
        for idx, v in enumerate(order):
            # Count neighbors already assigned to each side
            assigned = order[:idx]
            if len(assigned) == 0:
                continue
            
            side_0 = 0
            side_1 = 0
            for u in assigned:
                if adj[v, u]:
                    if labels[u] == 0:
                        side_0 += 1
                    else:
                        side_1 += 1
            
            # Assign to side with fewer neighbors (maximizes cut)
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
        
        # First: greedy local improvement
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # Then: simulated annealing to escape local optima
        # Use remaining time budget for this restart
        remaining_time = deadline - time.time()
        # Estimate steps based on time: ~1000 steps per second is a rough estimate
        est_steps = int(min(50000, max(1000, remaining_time * 800)))
        
        sa_labels, sa_cut = simulated_annealing(labels, max_steps=est_steps)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Final polish with local improvement
        if time.time() < deadline:
            polished = local_improve(sa_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished
    
    # If we have time left, do additional SA restarts from the best solution
    if time.time() < deadline and best_labels is not None:
        remaining = deadline - time.time()
        if remaining > 5.0:
            # A few more SA runs from the best found
            extra_steps = int(min(30000, max(2000, remaining * 500)))
            for extra in range(3):
                if time.time() > deadline:
                    break
                perturbed = best_labels.copy().astype(np.int8)
                # Perturb a few random vertices
                num_perturb = max(1, n // 10)
                perturbation_indices = rng.choice(n, size=num_perturb, replace=False)
                perturbed[perturbation_indices] = 1 - perturbed[perturbation_indices]
                
                sa_labels, sa_cut = simulated_annealing(perturbed, max_steps=extra_steps)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
                
                # Polish
                if time.time() < deadline:
                    polished = local_improve(sa_labels)
                    pol_cut = cut_size(polished)
                    if pol_cut > best_cut:
                        best_cut = pol_cut
                        best_labels = polished
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
