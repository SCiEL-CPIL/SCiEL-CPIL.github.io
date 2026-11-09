"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetric and remove self-loops
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Precompute total edge weight sum for potential normalization or checks
    # Energy E = sum_{i<j} A_{ij} s_i s_j. We minimize E.
    # Cut size C = 0.5 * (TotalEdges - E).
    # So minimizing E maximizes C.
    
    # Parallel Tempering (Replica Exchange Simulated Annealing)
    # This is "Quantum-Inspired" in the sense that it allows tunneling between 
    # local minima via state exchange between replicas at different temperatures.
    
    num_replicas = 10
    # Geometric temperature ladder
    T_min = 0.01
    T_max = 5.0
    temps = T_max * (T_min / T_max) ** (np.arange(num_replicas) / (num_replicas - 1))
    temps = temps[::-1]  # temps[0] is highest, temps[-1] is lowest
    
    # Initialize replicas
    # Use different random seeds for diversity
    rng = np.random.default_rng(42)
    replicas = []
    for i in range(num_replicas):
        # Random initialization
        s = rng.choice([-1, 1], size=n).astype(np.float64)
        replicas.append(s)
    
    # Function to compute energy for a replica
    def compute_energy(s):
        # E = sum_{i<j} A_{ij} s_i s_j
        # = 0.5 * (s^T A s - sum(A)) ... wait, diagonal is 0.
        # s^T A s = sum_i sum_j s_i A_ij s_j = 2 * sum_{i<j} A_ij s_i s_j
        # So E = 0.5 * s^T A s
        return 0.5 * s.dot(adj).dot(s)
    
    # Initialize energies
    energies = [compute_energy(s) for s in replicas]
    
    # Track best solution globally
    best_energy = min(energies)
    best_labels = replicas[np.argmin(energies)].copy()
    
    # SA Parameters
    # Total iterations per replica
    total_iters = 20000
    # Swap frequency: attempt swap every 'swap_interval' iterations
    swap_interval = 10
    
    for it in range(total_iters):
        # 1. Perform a move in each replica
        for r in range(num_replicas):
            s = replicas[r]
            E = energies[r]
            
            # Pick random vertex to flip
            v = rng.integers(0, n)
            
            # Calculate delta E for flipping v
            # E_new - E_old = 2 * s_v * sum_{j != v} A_vj s_j
            # Note: sum_{j != v} A_vj s_j is the v-th component of A s
            # Let's precompute A s for efficiency? Or just compute the dot product.
            # For N=100, dot product is fast.
            As = adj.dot(s)
            delta_E = 2.0 * s[v] * As[v]
            
            # Acceptance probability
            T = temps[r]
            if delta_E < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_E / T < 700: 
                    prob = np.exp(-delta_E / T)
                else:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                s[v] = -s[v]
                energies[r] += delta_E
                replicas[r] = s
                
                # Update global best
                if energies[r] < best_energy:
                    best_energy = energies[r]
                    best_labels = s.copy()
        
        # 2. Attempt swaps between adjacent replicas
        if (it + 1) % swap_interval == 0:
            for r in range(num_replicas - 1):
                r1 = r
                r2 = r + 1
                T1 = temps[r1]
                T2 = temps[r2]
                E1 = energies[r1]
                E2 = energies[r2]
                
                # Metropolis criterion for swap
                # Accept swap with prob exp(-(E2/T2 - E1/T1 - E1/T2 + E2/T1)) ?
                # Standard formula: P = exp( - (E2/T2 - E1/T1 - (E2-E1)/T_mid) ) ... 
                # Actually, the ratio of probabilities is:
                # P(swap) = min(1, exp( (E1/T1 - E2/T2) - (E2/T1 - E1/T2) ))
                # = min(1, exp( E1(1/T1 - 1/T2) - E2(1/T1 - 1/T2) ))
                # = min(1, exp( (E1 - E2) * (1/T1 - 1/T2) ))
                
                delta_swap = (E1 - E2) * (1.0/T1 - 1.0/T2)
                
                if delta_swap > 0:
                    # Swap is always accepted if it improves the "weight"
                    swap = True
                else:
                    if delta_swap > -700:
                        prob = np.exp(delta_swap)
                    else:
                        prob = 0.0
                    swap = rng.random() < prob
                
                if swap:
                    # Swap energies and states
                    energies[r1], energies[r2] = energies[r2], energies[r1]
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    
                    # Update global best
                    if energies[r1] < best_energy:
                        best_energy = energies[r1]
                        best_labels = replicas[r1].copy()
                    if energies[r2] < best_energy:
                        best_energy = energies[r2]
                        best_labels = replicas[r2].copy()
    
    # Final local search on the best replica found
    # Convert best_labels (spins) to labels (0/1)
    # s = 2x - 1 => x = (s + 1) / 2
    best_x = (best_labels + 1) // 2
    
    # One final polish with greedy single flips on the best solution
    s = best_labels.copy()
    E = compute_energy(s)
    improved = True
    while improved:
        improved = False
        As = adj.dot(s)
        for v in range(n):
            delta_E = 2.0 * s[v] * As[v]
            if delta_E < 0:
                s[v] = -s[v]
                E += delta_E
                improved = True
                # Update As? No, As depends on s. 
                # For N=100, recomputing As each iteration is fine, 
                # but we can just recompute As at start of loop.
        
    # The greedy loop above might be slow if it doesn't break. 
    # Let's do a simpler final polish: just one pass of best-improvement.
    s = best_labels.copy()
    for _ in range(5):
        improved = False
        As = adj.dot(s)
        for v in range(n):
            delta_E = 2.0 * s[v] * As[v]
            if delta_E < 0:
                s[v] = -s[v]
                improved = True
        if not improved:
            break
    
    final_labels = (s + 1) // 2
    
    return final_labels
# EVOLVE-BLOCK-END
