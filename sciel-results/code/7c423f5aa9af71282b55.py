"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Uses a coarse-to-fine grid search strategy, distinct from local minimization.
    """
    num_states = 2 ** n
    states = np.arange(num_states, dtype=np.int32)
    
    # Precompute cut values E for all states
    # E[s] is the number of edges cut by state s
    E = np.zeros(num_states, dtype=np.float64)
    for u, v in edges:
        # Vectorized bit extraction and comparison
        bu = (states >> u) & 1
        bv = (states >> v) & 1
        E += (bu != bv).astype(np.float64)
        
    # Precompute bit flip indices for each bit position to speed up mixing
    # flip_indices[i] is an array of length 2^n where flip_indices[i][s] = s ^ (1 << i)
    flip_indices = []
    for i in range(n):
        flip_indices.append(states ^ (1 << i))
        
    def simulate(g1, b1, g2, b2):
        """
        Simulate the QAOA circuit for p=2 and return the expected cut value.
        """
        # Initial state |+>^n
        c = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Layer 1: Cost
        c *= np.exp(-1j * g1 * E)
        
        # Layer 1: Mixing
        cb1 = np.cos(b1)
        sb1 = np.sin(b1)
        for i in range(n):
            # Apply exp(-i b1 X_i)
            # new_c = cos(b1) * c - i * sin(b1) * c[flip_i]
            c = cb1 * c - 1j * sb1 * c[flip_indices[i]]
            
        # Layer 2: Cost
        c *= np.exp(-1j * g2 * E)
        
        # Layer 2: Mixing
        cb2 = np.cos(b2)
        sb2 = np.sin(b2)
        for i in range(n):
            c = cb2 * c - 1j * sb2 * c[flip_indices[i]]
            
        # Expectation value
        probs = np.abs(c) ** 2
        return np.sum(probs * E)

    # Grid Search Parameters
    # Angles are periodic with 2*pi. Due to symmetry, we can restrict to [0, pi] or similar.
    # For p=2, optimal angles are often in [0, pi/2] or [0, pi].
    # Let's search in [0, pi] for all 4 angles.
    
    # Coarse Grid
    coarse_steps = 10
    coarse_range = np.linspace(0, np.pi, coarse_steps)
    
    best_val = -np.inf
    best_params = (0, 0, 0, 0)
    
    # 4D Grid Search
    # To avoid huge memory allocation, we loop
    # 10^4 = 10,000 evaluations. Each is ~10k ops. Total 100M ops. Might be slow in Python loop.
    # Let's reduce coarse steps to 5. 5^4 = 625 evals. Fast.
    coarse_steps = 5
    coarse_range = np.linspace(0, np.pi, coarse_steps)
    
    for g1 in coarse_range:
        for b1 in coarse_range:
            for g2 in coarse_range:
                for b2 in coarse_range:
                    val = simulate(g1, b1, g2, b2)
                    if val > best_val:
                        best_val = val
                        best_params = (g1, b1, g2, b2)
                        
    # Fine Grid around best_params
    fine_steps = 11
    margin = np.pi / coarse_steps # Half the coarse step size
    g1_min = max(0, best_params[0] - margin)
    g1_max = min(np.pi, best_params[0] + margin)
    b1_min = max(0, best_params[1] - margin)
    b1_max = min(np.pi, best_params[1] + margin)
    g2_min = max(0, best_params[2] - margin)
    g2_max = min(np.pi, best_params[2] + margin)
    b2_min = max(0, best_params[3] - margin)
    b2_max = min(np.pi, best_params[3] + margin)
    
    fine_g1 = np.linspace(g1_min, g1_max, fine_steps)
    fine_b1 = np.linspace(b1_min, b1_max, fine_steps)
    fine_g2 = np.linspace(g2_min, g2_max, fine_steps)
    fine_b2 = np.linspace(b2_min, b2_max, fine_steps)
    
    for g1 in fine_g1:
        for b1 in fine_b1:
            for g2 in fine_g2:
                for b2 in fine_b2:
                    val = simulate(g1, b1, g2, b2)
                    if val > best_val:
                        best_val = val
                        best_params = (g1, b1, g2, b2)
                        
    # Even finer grid?
    # 11^4 = 14641 evals. Each ~10k ops. 146M ops. Might be borderline for 120s in Python if not optimized.
    # Let's stick to 2 levels. The coarse grid finds the basin, fine grid refines.
    
    g1, b1, g2, b2 = best_params
    
    return np.array([
        [g1, b1],
        [g2, b2]
    ])

# EVOLVE-BLOCK-END
