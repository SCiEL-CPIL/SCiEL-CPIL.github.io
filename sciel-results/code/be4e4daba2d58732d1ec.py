import numpy as np
import time

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _simulate_energy(params, n, layers, H):
    """
    Vectorized simulation of the circuit to compute the energy.
    params: (layers, n)
    H: Hamiltonian matrix
    Returns: energy (float)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Precompute masks for speed
    # We will apply gates in order.
    # Ry(theta) on qubit q:
    # |0> -> c|0> + s|1>
    # |1> -> -s|0> + c|1>
    # Vectorized:
    # For a given qubit q, we split the state into components where bit q is 0 and 1.
    # indices_0 = [b for b in range(dim) if (b & (1<<q)) == 0]
    # indices_1 = [b for b in range(dim) if (b & (1<<q)) != 0]
    # state_new[indices_0] = c * state[indices_0] - s * state[indices_1]
    # state_new[indices_1] = s * state[indices_0] + c * state[indices_1]
    
    # Precompute index pairs for each qubit
    index_pairs = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([b for b in range(dim) if (b & mask) == 0])
        idx1 = idx0 + mask
        index_pairs.append((idx0, idx1))
        
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = index_pairs[q]
            
            # We need to be careful with in-place updates or using a new array.
            # Since qubits are applied sequentially, we must update the state.
            # state[idx0] and state[idx1] are updated based on their old values.
            old_0 = state[idx0].copy()
            old_1 = state[idx1].copy()
            
            state[idx0] = c * old_0 - s * old_1
            state[idx1] = s * old_0 + c * old_1
            
        # Apply CZ between layers (except after last)
        if l < layers - 1:
            for q in range(n - 1):
                q1, q2 = q, q + 1
                mask1 = 1 << q1
                mask2 = 1 << q2
                # CZ flips sign if both bits are 1.
                # Indices where both are 1: (b & mask1) != 0 AND (b & mask2) != 0
                # We can find these efficiently.
                # b has bit q1 and bit q2 set.
                # Let's just iterate or use boolean mask.
                # Boolean mask is fast enough for dim <= 64.
                mask_bool = ((np.arange(dim) & mask1) != 0) & ((np.arange(dim) & mask2) != 0)
                state[mask_bool] *= -1
                
    # Compute Energy
    Hpsi = H @ state
    energy = np.vdot(state, Hpsi).real
    return energy

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _ground_energy(n, coupling, field)
    
    # Simulated Annealing
    # Parameters: (layers, n)
    # Objective: minimize E_circuit - E0 (or just E_circuit since E0 is constant)
    
    # Initial parameters
    # Heuristic: For transverse field Ising, ground state is often close to |0> in Z basis 
    # if field is small, or |+> in X basis (pi/2 in Y) if field is large.
    # Let's start with a mix or random.
    
    best_params = None
    best_energy = np.inf
    current_params = None
    current_energy = np.inf
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.995 # Cooling rate
    max_steps = 50000
    
    # To ensure we explore well, we can run multiple restarts if time permits, 
    # or just one long SA with periodic reheating.
    # Given the small dimension (18 params), a single long SA might be enough, 
    # but multiple short ones might be safer.
    # Let's do a main SA run, and if time remains, do local polish.
    
    def run_sa(params_init, steps, T0, T1):
        params = params_init.copy()
        energy = _simulate_energy(params, n, layers, H)
        best_p = params.copy()
        best_e = energy
        T = T0
        
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Perturb a random parameter
            l = np.random.randint(layers)
            q = np.random.randint(n)
            delta = np.random.normal(0, 0.1)
            new_params = params.copy()
            new_params[l, q] += delta
            
            new_energy = _simulate_energy(new_params, n, layers, H)
            
            # Acceptance criterion
            if new_energy < energy:
                params = new_params
                energy = new_energy
                if energy < best_e:
                    best_e = energy
                    best_p = params.copy()
            else:
                # Metropolis criterion
                if T > 0:
                    prob = np.exp(-(new_energy - energy) / T)
                    if np.random.random() < prob:
                        params = new_params
                        energy = new_energy
                        
            # Cool
            T *= alpha
            if T < T1:
                T = T1
                
        return best_p, best_e

    # Run 1: Main SA
    # Initialize with a "physical" guess: 
    # If field is large, pi/2 is good. If small, 0 is good.
    # Let's try a few inits.
    
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi/2),
        np.random.uniform(-np.pi, np.pi, (layers, n))
    ]
    
    # We will distribute time among a few SA runs with different inits
    # Since SA is stochastic, running it once for a long time is often better than many short ones,
    # but different inits help avoid local minima.
    # Let's do 3 runs, each with 1/3 of the time budget.
    
    time_per_run = (deadline - time.time()) / 3.0
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Adjust steps based on time available
        # Estimate time per step
        t_start = time.time()
        _ = _simulate_energy(init, n, layers, H)
        t_step = time.time() - t_start
        
        if t_step < 1e-9: t_step = 1e-9
        steps = int(time_per_run / t_step)
        if steps < 100: steps = 100
        
        p, e = run_sa(init, steps, T_start, T_end)
        
        if e < best_energy:
            best_energy = e
            best_params = p
            
    # Local Polish: Use Nelder-Mead or Gradient Descent on the best result found
    if best_params is not None and time.time() < deadline - 5.0:
        # Simple Gradient Descent polish
        # We need gradients. Let's use finite differences for simplicity in this block,
        # or just a few steps of Nelder-Mead.
        # Nelder-Mead is robust.
        from scipy.optimize import minimize
        
        def objective(p):
            p_reshaped = p.reshape(layers, n)
            return _simulate_energy(p_reshaped, n, layers, H)
            
        try:
            res = minimize(objective, best_params.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.reshape(layers, n)
        except Exception:
            pass

    if best_params is None:
        # Fallback
        return np.full((layers, n), np.pi/2)
        
    return best_params
