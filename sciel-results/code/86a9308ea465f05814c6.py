"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_overlaps(h):
    """Compute all circular overlaps of h with shifted (1-h)."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H)**2).real
    overlaps = N/2 - autocorr
    return overlaps

def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    return np.max(_compute_overlaps(h))

def _project_to_valid(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    for _ in range(100):
        s = np.sum(h)
        diff = N/2 - s
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] += diff * (1 - h[mask]) / np.sum(1 - h[mask])
        else:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] += diff * h[mask] / np.sum(h[mask])
        h = np.clip(h, 0, 1)
    # Exact fix
    diff = N/2 - np.sum(h)
    if abs(diff) > 1e-12:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0, 1)
    return h

def _ising_energy(spin, h, k_max):
    """
    Energy of a spin flip at position i in the Ising model.
    spin[i] = +1 means h[i] is closer to 1, -1 means closer to 0.
    Antiferromagnetic coupling favors alternating spins.
    The mean-field term enforces sum constraint.
    """
    i = spin  # not used directly, just for clarity
    # Local field from antiferromagnetic coupling to neighbors
    # J * sum_j J_ij * s_j where J_ij = -1 for neighbors (antiferro)
    # Simplified: energy change from flipping s[i]
    # E = -J * sum_neighbors s_j * s_i + h_mean_field * s_i
    # For antiferro: J > 0, coupling = -J*s_i*s_j
    # Flipping s_i changes energy by: 2*J*s_i*(sum_neighbors s_j) - 2*h_mf*s_i
    
    # Actually, let's define the energy more carefully.
    # We want to minimize max overlap. The overlap for shift k is:
    # overlap_k = sum_i h[i]*(1-h[(i+k)%N])
    # If we think of h[i] as determined by spin[i], with h[i] = (1+spin[i])/2 * a + (1-spin[i])/2 * b
    # where a > b are the high and low values.
    
    # Simpler approach: use the overlap directly as the "energy"
    # and do Glauber dynamics on the spin configuration.
    
    # For now, compute the change in max overlap if we flip spin[i]
    return 0  # placeholder, we'll compute differently

def _spin_to_h(spin, high_val=0.8, low_val=0.2):
    """Convert spin configuration to step heights."""
    h = np.where(spin > 0, high_val, low_val)
    return _project_to_valid(h)

def _h_to_spin(h):
    """Convert step heights to spin configuration."""
    return np.where(h >= 0.5, 1, -1).astype(float)

def _glauber_step(spin, h, T, rng):
    """
    Perform one Glauber dynamics step on the Ising chain.
    Returns new spin and whether it was accepted.
    """
    # Pick a random site
    i = rng.integers(0, N)
    
    # Compute local field
    # Antiferromagnetic coupling: neighbors contribute -J*s_neighbor
    J = 1.0  # coupling strength
    # Neighbors in circular chain
    left = (i - 1) % N
    right = (i + 1) % N
    
    # Local field from Ising interactions
    h_local = -J * (spin[left] + spin[right])
    
    # Mean-field term to enforce sum constraint
    # If too many +1 spins, add negative field to discourage more +1
    num_plus = np.sum(spin > 0)
    target_plus = N / 2
    # Mean field pushes toward target fraction
    h_mf = 0.1 * (num_plus - target_plus)
    # If num_plus > target, h_mf > 0, which favors flipping +1 to -1
    # In Glauber: P(flip) = 1/(1+exp(-s_i * (h_local + h_mf) / T))
    # If s_i = +1 and h_local + h_mf > 0, less likely to flip (stays +1)
    # If s_i = -1 and h_local + h_mf > 0, more likely to flip (becomes +1)
    
    # Effective field
    eff_field = h_local + h_mf
    
    # Glauber acceptance probability
    # P(s_i -> -s_i) = 1 / (1 + exp(-s_i * eff_field / T))
    # If s_i = +1: P(flip to -1) = 1/(1+exp(-eff_field/T))
    # If s_i = -1: P(flip to +1) = 1/(1+exp(eff_field/T))
    
    if spin[i] > 0:
        p_accept = 1.0 / (1.0 + np.exp(-eff_field / T))
    else:
        p_accept = 1.0 / (1.0 + np.exp(eff_field / T))
    
    if rng.random() < p_accept:
        new_spin = spin.copy()
        new_spin[i] = -spin[i]
        return new_spin, True
    return spin, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating (perfect antiferromagnetic)
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    h_alt[1::2] = 0.0
    candidates.append(h_alt)
    
    # Sinusoidal patterns
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 13, 17, 23, 31]:
        h = 0.5 + 0.5 * np.cos(freq * x)
        candidates.append(_project_to_valid(h))
    
    # Random
    for _ in range(30):
        h_rand = rng.random(N)
        candidates.append(_project_to_valid(h_rand))
    
    # Evaluate all candidates
    best_h = None
    best_score = float('inf')
    for h in candidates:
        s = _max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Glauber dynamics (Ising spin chain optimization)
    # Initialize spin from best_h
    spin = _h_to_spin(best_h)
    h_curr = _spin_to_h(spin)
    s_curr = _max_overlap(h_curr)
    
    if s_curr < best_score:
        best_score = s_curr
        best_h = h_curr.copy()
    
    # Simulated annealing with Glauber dynamics
    T_start = 1.0
    T_end = 0.01
    n_sweeps = 500
    sweep_size = N  # each sweep visits each site once
    
    for sweep in range(n_sweeps):
        if time.time() > deadline - 20:
            break
        
        # Annealing schedule
        progress = sweep / n_sweeps
        T = T_start * (T_end / T_start) ** progress
        
        # One full sweep: visit each site once
        sites = rng.permutation(N)
        for i in sites:
            if time.time() > deadline - 15:
                break
            
            # Compute local field
            J = 1.0
            left = (i - 1) % N
            right = (i + 1) % N
            h_local = -J * (spin[left] + spin[right])
            
            # Mean-field term for sum constraint
            num_plus = np.sum(spin > 0)
            h_mf = 0.2 * (num_plus - N/2)
            
            eff_field = h_local + h_mf
            
            # Glauber acceptance
            if spin[i] > 0:
                p_accept = 1.0 / (1.0 + np.exp(-eff_field / max(T, 1e-8)))
            else:
                p_accept = 1.0 / (1.0 + np.exp(eff_field / max(T, 1e-8)))
            
            if rng.random() < p_accept:
                spin[i] = -spin[i]
        
        # After each sweep, convert to h and evaluate
        h_new = _spin_to_h(spin)
        s_new = _max_overlap(h_new)
        
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
    
    # Phase 2: Local refinement using subgradient descent
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        lr = 0.05
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            overlaps = _compute_overlaps(h_curr)
            k_max = np.argmax(overlaps)
            
            # Subgradient
            grad = np.zeros(N)
            for j in range(N):
                grad[j] = (1 - h_curr[(j + k_max) % N]) - h_curr[(j - k_max) % N]
            
            grad = grad - np.mean(grad)
            
            h_curr = h_curr - lr * grad
            h_curr = _project_to_valid(h_curr)
            
            s = _max_overlap(h_curr)
            if s < best_score:
                best_score = s
                best_h = h_curr.copy()
            
            if step % 200 == 0 and step > 0:
                lr *= 0.9
    
    # Phase 3: Random restarts with short Glauber runs
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        n_restarts = int(remaining_time / 2.0)
        n_restarts = max(1, min(n_restarts, 20))
        
        for r in range(n_restarts):
            if time.time() > deadline - 3:
                break
            
            # Random initial spin
            spin_r = rng.choice([-1, 1], size=N).astype(float)
            h_r = _spin_to_h(spin_r)
            s_r = _max_overlap(h_r)
            
            # Short Glauber run
            T_r = 0.5
            n_short_sweeps = 50
            for sweep in range(n_short_sweeps):
                if time.time() > deadline - 2:
                    break
                sites = rng.permutation(N)
                for i in sites:
                    J = 1.0
                    left = (i - 1) % N
                    right = (i + 1) % N
                    h_local = -J * (spin_r[left] + spin_r[right])
                    num_plus = np.sum(spin_r > 0)
                    h_mf = 0.2 * (num_plus - N/2)
                    eff_field = h_local + h_mf
                    
                    if spin_r[i] > 0:
                        p_acc = 1.0 / (1.0 + np.exp(-eff_field / T_r))
                    else:
                        p_acc = 1.0 / (1.0 + np.exp(eff_field / T_r))
                    
                    if rng.random() < p_acc:
                        spin_r[i] = -spin_r[i]
                
                h_r = _spin_to_h(spin_r)
                s_r = _max_overlap(h_r)
                
                if s_r < best_score:
                    best_score = s_r
                    best_h = h_r.copy()
    
    h_final = _project_to_valid(best_h)
    return h_final
# EVOLVE-BLOCK-END
