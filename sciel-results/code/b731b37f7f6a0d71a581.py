"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Goemans-Williamson SDP Relaxation, Randomized Rounding,
    Simulated Annealing, and Enhanced Local Search.
    
    1. SDP Relaxation via Projected Gradient Descent on V:
       - Directly optimize V (n x n) with unit row norms.
       - X = V V^T is inherently PSD, so no eigendecomposition projection is needed.
       - This is significantly faster than X-based methods with PSD projection.
       
    2. Randomized Hyperplane Rounding:
       - Generate diverse initial partitions from the SDP solution.
       
    3. Simulated Annealing:
       - Tuned for better exploration with higher initial temperature.
       
    4. Enhanced Local Polish:
       - Multi-pass single-flip optimization.
       - Pairwise swap optimization to escape 1-flip local optima.
    """
    start_time = time.time()
    # Safety margin: 150s limit, keep 5s buffer
    deadline = start_time + 145.0
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Helper to compute cut size
    def calc_cut(s):
        total_edges = 0.5 * np.sum(adj)
        val = s @ adj @ s
        return total_edges - 0.5 * val
    
    # 1. SDP Relaxation via Projected Gradient Descent on V
    # Objective: maximize 0.25 * sum_{ij} A_ij * (1 - V_i^T V_j)
    # Gradient w.r.t V_i: -0.25 * sum_j A_ij V_j = -0.25 * (A @ V)_i
    
    rng_init = np.random.default_rng(42)
    V = rng_init.standard_normal((n, n))
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms[row_norms < 1e-10] = 1e-10
    V = V / row_norms
    
    # SDP Parameters
    # We can afford more iterations now since we removed the expensive eigendecomposition
    sdp_iters = 500 
    lr = 0.1  # Higher learning rate as we are not doing expensive projections
    
    for it in range(sdp_iters):
        if time.time() > deadline - 100:  # Leave ample time for SA and polish
            break
        
        # Gradient step
        grad_V = -0.25 * (adj @ V)
        V = V + lr * grad_V
        
        # Project to unit norm (simple and fast)
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        row_norms[row_norms < 1e-10] = 1e-10
        V = V / row_norms
    
    # 2. Randomized Hyperplane Rounding
    rng = np.random.default_rng()
    
    def generate_rounding(V, num_rounds=10):
        """Generate multiple partitions using random hyperplane rounding."""
        partitions = []
        for _ in range(num_rounds):
            # Random unit vector
            r = rng.standard_normal(n)
            r /= (np.linalg.norm(r) + 1e-10)
            # Round: sign(V @ r)
            s = np.sign(V @ r)
            s[s == 0] = 1  # Handle zeros
            partitions.append(s)
        return partitions
    
    # Generate diverse initial partitions
    initial_partitions = generate_rounding(V, num_rounds=20)
    
    # Add spectral initialization
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        idx_min = np.argmin(eigenvalues)
        u = eigenvectors[:, idx_min]
        s_spec = (u >= 0).astype(np.float64) * 2 - 1
        initial_partitions.append(s_spec)
    except Exception:
        pass
    
    # Add random initializations
    for _ in range(5):
        s_rand = rng.choice([-1.0, 1.0], size=n)
        initial_partitions.append(s_rand)
    
    # 3. Simulated Annealing with adaptive cooling
    best_cut = -1
    best_s = None
    
    T0 = 30.0  # Higher initial temperature for better exploration
    T_min = 0.05
    cooling_rate = 0.997  # Slower cooling for more thorough search
    
    for init_s in initial_partitions:
        if time.time() > deadline - 10:
            break
        
        s = init_s.copy()
        labels = ((s + 1) / 2).astype(np.int8)
        As = adj @ s
        current_cut = calc_cut(s)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_s = s.copy()
        
        T = T0
        no_improve_count = 0
        max_no_improve = 500
        
        while T > T_min and no_improve_count < max_no_improve:
            if time.time() > deadline - 10:
                break
            
            # Randomly select a vertex to flip
            v = rng.integers(n)
            
            # Calculate delta for flipping v
            delta = 2 * s[v] * As[v]
            
            # Accept if improving, or with probability exp(delta/T) if worsening
            if delta >= 0:
                accept = True
            else:
                # Avoid underflow
                if delta / T < -100:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
            
            if accept:
                # Apply flip
                old_s_v = s[v]
                s[v] = -old_s_v
                labels[v] = 1 - labels[v]
                
                # Update As
                As -= 2 * old_s_v * adj[v]
                
                # Update current cut
                current_cut += delta
                
                # Update best
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_s = s.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            
            # Occasionally do a local polish to escape local optima
            if rng.random() < 0.03:
                # Quick local polish: flip any vertex with positive delta
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    for v in range(n):
                        delta = 2 * s[v] * As[v]
                        if delta > 1e-10:
                            old_s_v = s[v]
                            s[v] = -old_s_v
                            labels[v] = 1 - labels[v]
                            As -= 2 * old_s_v * adj[v]
                            current_cut += delta
                            improved = True
                            if current_cut > best_cut:
                                best_cut = current_cut
                                best_s = s.copy()
                                no_improve_count = 0
        
        # Final local polish for this partition
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in range(n):
                delta = 2 * s[v] * As[v]
                if delta > 1e-10:
                    old_s_v = s[v]
                    s[v] = -old_s_v
                    labels[v] = 1 - labels[v]
                    As -= 2 * old_s_v * adj[v]
                    current_cut += delta
                    improved = True
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_s = s.copy()
    
    # 4. Final High-Precision Local Polish on the Best Solution
    if best_s is not None:
        s = best_s.copy()
        labels = ((s + 1) / 2).astype(np.int8)
        As = adj @ s
        current_cut = calc_cut(s)
        
        # Phase 1: Single-flip polish (tight tolerance)
        for polish_pass in range(10):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for v in range(n):
                    delta = 2 * s[v] * As[v]
                    if delta > 1e-10:
                        old_s_v = s[v]
                        s[v] = -old_s_v
                        labels[v] = 1 - labels[v]
                        As -= 2 * old_s_v * adj[v]
                        current_cut += delta
                        improved = True
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_s = s.copy()
        
        # Phase 2: Pairwise Swap Polish
        # For n=100, checking all pairs is 4950 checks, which is very fast.
        # This helps escape local optima that are stable under single flips.
        if time.time() < deadline - 2:
            improved = True
            # Limit number of full pair-swap passes to avoid wasting time
            for swap_pass in range(20):
                if time.time() > deadline - 2:
                    break
                improved = False
                for i in range(n):
                    for j in range(i + 1, n):
                        # Delta for swapping i and j
                        # Flipping i: delta_i = 2 * s[i] * As[i]
                        # Flipping j: delta_j = 2 * s[j] * As[j]
                        # Interaction term: -4 * adj[i,j] * s[i] * s[j]
                        delta_i = 2 * s[i] * As[i]
                        delta_j = 2 * s[j] * As[j]
                        delta_swap = delta_i + delta_j - 4 * adj[i, j] * s[i] * s[j]
                        
                        if delta_swap > 1e-10:
                            old_si = s[i]
                            old_sj = s[j]
                            s[i] = -old_si
                            s[j] = -old_sj
                            labels[i] = 1 - labels[i]
                            labels[j] = 1 - labels[j]
                            
                            # Update As for all vertices affected by i and j flips
                            # As = A @ s. 
                            # Change in s: -2*old_si at i, -2*old_sj at j
                            # As_k = sum_m A_km s_m
                            # Delta As_k = A_ki * (-2*old_si) + A_kj * (-2*old_sj)
                            As -= 2 * old_si * adj[:, i]
                            As -= 2 * old_sj * adj[:, j]
                            
                            current_cut += delta_swap
                            improved = True
                            if current_cut > best_cut:
                                best_cut = current_cut
                                best_s = s.copy()
                
                if not improved:
                    break
        
        # Phase 3: Triple-flip polish for harder-to-escape local optima
        if time.time() < deadline - 2:
            # Sample random triples to check
            for triple_pass in range(50):
                if time.time() > deadline - 2:
                    break
                # Sample 100 random triples
                for _ in range(100):
                    i, j, k = rng.choice(n, size=3, replace=False)
                    if i == j or j == k or i == k:
                        continue
                    
                    # Calculate delta for flipping i, j, k simultaneously
                    delta_i = 2 * s[i] * As[i]
                    delta_j = 2 * s[j] * As[j]
                    delta_k = 2 * s[k] * As[k]
                    
                    # Interaction terms
                    interaction = (4 * adj[i, j] * s[i] * s[j] + 
                                   4 * adj[i, k] * s[i] * s[k] + 
                                   4 * adj[j, k] * s[j] * s[k])
                    
                    delta_triple = delta_i + delta_j + delta_k - interaction
                    
                    if delta_triple > 1e-10:
                        old_si = s[i]
                        old_sj = s[j]
                        old_sk = s[k]
                        s[i] = -old_si
                        s[j] = -old_sj
                        s[k] = -old_sk
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        labels[k] = 1 - labels[k]
                        
                        # Update As
                        As -= 2 * old_si * adj[:, i]
                        As -= 2 * old_sj * adj[:, j]
                        As -= 2 * old_sk * adj[:, k]
                        
                        current_cut += delta_triple
                        
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_s = s.copy()
    
    # Return best labels found
    best_labels = ((best_s + 1) / 2).astype(np.int8)
    return best_labels
# EVOLVE-BLOCK-END
