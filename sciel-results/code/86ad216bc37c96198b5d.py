"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Ensure symmetric
    adj = np.maximum(adj, adj.T)
    # Remove self-loops if any
    np.fill_diagonal(adj, 0)
    
    # Precompute neighbor lists for efficient updates
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    best_labels_global = None
    best_cut_global = -1
    
    # SA Parameters
    num_restarts = 5
    time_per_restart = 20.0
    
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        # Initialize spins randomly
        spins = rng.choice([-1, 1], size=n)
        
        # Precompute field h_v = sum_u A_vu s_u
        h = adj @ spins
        
        # Calculate initial energy
        current_energy = 0.5 * np.dot(spins, h)
        
        # Track best
        best_energy_local = current_energy
        best_spins_local = spins.copy()
        
        # SA Schedule
        T_start = 50.0
        T_end = 0.1
        iterations = 2000000
        cool_factor = (T_end / T_start) ** (1.0 / iterations)
        
        start_time = time.time()
        
        for i in range(iterations):
            if (i % 1000 == 0) and (time.time() - start_time > time_per_restart):
                break
            
            # Pick random vertex
            v = rng.integers(0, n)
            
            # Calculate delta E for flipping v
            delta_e = -2 * spins[v] * h[v]
            
            if delta_e < 0:
                accept = True
            else:
                T = T_start * (cool_factor ** i)
                accept = rng.random() < np.exp(-delta_e / T)
                
            if accept:
                # Flip spin v
                old_spin = spins[v]
                spins[v] = -old_spin
                current_energy += delta_e
                
                # Update h for neighbors of v
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    h[nbrs] -= 2 * adj[v, nbrs] * old_spin
                
                if current_energy < best_energy_local:
                    best_energy_local = current_energy
                    best_spins_local = spins.copy()
        
        # Convert best spins to labels
        labels = (best_spins_local + 1) // 2
        
        # Calculate cut size to compare globally
        mask = (labels[:, None] != labels[None, :])
        cut_size = np.triu(adj * mask, k=1).sum()
        
        if cut_size > best_cut_global:
            best_cut_global = cut_size
            best_labels_global = labels.copy()
            
    return best_labels_global
# EVOLVE-BLOCK-END
