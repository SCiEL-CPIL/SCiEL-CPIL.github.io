"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(x):
    """
    Compute the C3 upper bound for the given sequence x (binary +/-1).
    C3 = max_{k != 0} |R(k)| / N where R(k) is the linear autocorrelation.
    """
    N_local = len(x)
    size = 1
    while size < 2 * N_local:
        size *= 2
    
    fft_x = np.fft.fft(x, size)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real
    
    sidelobes = acf[1:N_local]
    max_sidelobe = np.max(np.abs(sidelobes))
    
    return max_sidelobe / N_local

def _compute_c3_bound_complex(z):
    """
    Compute a continuous surrogate for C3 using complex unit-modulus vector z.
    Uses |R(k)|^2 / N^2 as a smooth proxy, taking the soft-max over lags.
    """
    N_local = len(z)
    size = 1
    while size < 2 * N_local:
        size *= 2
    
    fft_z = np.fft.fft(z, size)
    acf = np.fft.ifft(fft_z * np.conj(fft_z)).real
    
    sidelobes = acf[1:N_local]
    # Use squared magnitude to make it differentiable
    energies = sidelobes ** 2
    # Soft-max approximation of max
    # Use a high temperature for sharpness
    t = 100.0
    log_sum_exp = np.log(np.sum(np.exp(energies / t))) * t
    # This is an upper bound on max(energies)
    # Return sqrt to approximate max|sidelobe|
    max_energy = np.sqrt(log_sum_exp)
    
    return max_energy / N_local

def _phase_gradient_descent(z0, max_iters=5000, lr=0.05, time_limit=10.0):
    """
    Minimize the continuous surrogate C3 on the unit-modulus manifold.
    z is a complex vector where |z[i]| = 1.
    We update phases: z_new = z * exp(i * gradient)
    """
    N_local = len(z0)
    z = z0.astype(np.complex128).copy()
    # Ensure unit modulus
    z /= np.abs(z) + 1e-12
    
    size = 1
    while size < 2 * N_local:
        size *= 2
    
    start_time = time.time()
    current_cost = _compute_c3_bound_complex(z)
    
    # Precompute FFT size
    fft_size = size
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute autocorrelation
        fft_z = np.fft.fft(z, fft_size)
        acf = np.fft.ifft(fft_z * np.conj(fft_z)).real
        
        # Gradient of max |R(k)|^2 w.r.t. z
        # R(k) = sum_{i=0}^{N-1-k} z[i] * conj(z[i+k])
        # d/dz[i] |R(k)|^2 = 2 * R(k) * conj(dR(k)/dz[i])
        # dR(k)/dz[i] = conj(z[i+k]) if i < N-k, else 0
        # So d/dz[i] |R(k)|^2 = 2 * R(k) * z[i+k] (for i < N-k)
        
        # But we're using soft-max, so we need weighted gradient
        # Let's use the gradient of the soft-max directly
        # Soft-max S = t * log(sum(exp(R(k)^2 / t)))
        # dS/dR(k)^2 = exp(R(k)^2 / t) / sum(exp(R(k)^2 / t))
        # dS/dz[i] = sum_k weights[k] * d(R(k)^2)/dz[i]
        
        sidelobes = acf[1:N_local]
        energies = sidelobes ** 2
        t = 100.0
        exp_energies = np.exp(energies / t)
        sum_exp = np.sum(exp_energies)
        weights = exp_energies / sum_exp
        
        # Compute gradient w.r.t. z
        # grad_z[i] = sum_k weights[k] * 2 * R(k) * conj(conj(z[i+k]))
        #           = sum_k weights[k] * 2 * R(k) * z[i+k]  (for i < N-1-k)
        # Also for the conjugate part:
        # R(k) = sum_{i} z[i] * conj(z[i+k])
        # d/dconj(z[i]) R(k) = z[i+k] (if i+k < N)
        # So d| R(k) |^2 / dconj(z[i]) = 2 * R(k) * z[i+k] (if i+k < N)
        
        # The gradient descent on the phase manifold:
        # We want to minimize S(z). 
        # z = exp(i*theta)
        # dS/dtheta[i] = Im(conj(z[i]) * dS/dz[i])
        
        # Let's compute dS/dz[i]
        # dS/dz[i] = sum_k weights[k] * 2 * conj(R(k)) * conj(z[i+k])  (for i+k < N)
        #           + sum_k weights[k] * 2 * R(k) * conj(z[i-k])     (for i-k >= 0)
        # Wait, let's be more careful.
        
        # R(k) = sum_{j=0}^{N-1-k} z[j] * conj(z[j+k])
        # dR(k)/dz[i] = conj(z[i+k]) if i < N-k
        # dR(k)/dconj(z[i]) = z[i+k] if i < N-k
        
        # |R(k)|^2 = R(k) * conj(R(k))
        # d|R(k)|^2 / dz[i] = conj(R(k)) * dR(k)/dz[i] + R(k) * dconj(R(k))/dz[i]
        # dconj(R(k))/dz[i] = 0 (since conj(R(k)) depends on conj(z))
        # So d|R(k)|^2 / dz[i] = conj(R(k)) * conj(z[i+k]) (if i < N-k)
        
        # d|R(k)|^2 / dconj(z[i]) = R(k) * z[i+k] (if i < N-k)
        
        # For gradient descent on theta:
        # z = exp(i*theta)
        # dz/dtheta = i*z
        # dS/dtheta[i] = Re(dS/dz[i] * conj(dz/dtheta)) + Re(dS/dconj(z[i]) * conj(dconj(z)/dtheta))
        # conj(dz/dtheta) = conj(i*z) = -i*conj(z)
        # dconj(z)/dtheta = -i*conj(z)
        # conj(dconj(z)/dtheta) = i*z
        
        # dS/dtheta[i] = Re(dS/dz[i] * (-i*conj(z[i]))) + Re(dS/dconj(z[i]) * (i*z[i]))
        #              = Re(-i * dS/dz[i] * conj(z[i])) + Re(i * dS/dconj(z[i]) * z[i])
        
        # Let's compute dS/dz and dS/dconj(z)
        dS_dz = np.zeros(N_local, dtype=np.complex128)
        dS_dconj_z = np.zeros(N_local, dtype=np.complex128)
        
        for k in range(N_local - 1):
            w = weights[k]
            Rk = sidelobes[k]
            if abs(Rk) < 1e-10:
                continue
            # d|R(k)|^2 / dz[i] = conj(R(k)) * conj(z[i+k]) for i = 0..N-1-k-1
            # d|R(k)|^2 / dconj(z[i]) = R(k) * z[i+k] for i = 0..N-1-k-1
            # Also need to account for the conjugate part in |R(k)|^2
            # Actually, |R(k)|^2 = R(k) * conj(R(k))
            # d/dz[i] (R(k) * conj(R(k))) = (dR(k)/dz[i]) * conj(R(k)) + R(k) * (dconj(R(k))/dz[i])
            # dconj(R(k))/dz[i] = 0
            # So d|R(k)|^2/dz[i] = conj(R(k)) * dR(k)/dz[i] = conj(R(k)) * conj(z[i+k])
            
            # d/dconj(z[i]) (R(k) * conj(R(k))) = (dR(k)/dconj(z[i])) * conj(R(k)) + R(k) * (dconj(R(k))/dconj(z[i]))
            # dR(k)/dconj(z[i]) = 0
            # dconj(R(k))/dconj(z[i]) = conj(dR(k)/dz[i]) = conj(conj(z[i+k])) = z[i+k]
            # So d|R(k)|^2/dconj(z[i]) = R(k) * z[i+k]
            
            if N_local - 1 - k > 0:
                dS_dz[:N_local-1-k] += w * np.conj(Rk) * np.conj(z[k:N_local])
                dS_dconj_z[:N_local-1-k] += w * Rk * z[k:N_local]
        
        # Now compute dS/dtheta
        # dS/dtheta[i] = Re(-i * dS_dz[i] * conj(z[i])) + Re(i * dS_dconj_z[i] * z[i])
        dS_dtheta = np.zeros(N_local, dtype=np.float64)
        dS_dtheta = np.real(-1j * dS_dz * np.conj(z)) + np.real(1j * dS_dconj_z * z)
        
        # Gradient descent step
        z = z * np.exp(-1j * lr * dS_dtheta)
        # Re-normalize to unit modulus
        z /= np.abs(z) + 1e-12
        
        new_cost = _compute_c3_bound_complex(z)
        if new_cost < current_cost:
            current_cost = new_cost
        else:
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-6:
                break
    
    # Return the best z
    return z, current_cost

def _phase_to_binary(z):
    """Convert complex unit-modulus vector to binary +/-1 based on real part sign."""
    return np.sign(np.real(z))
    # If real part is 0, use imaginary part
    # But for now, sign of real should be fine

def _discrete_local_search(x, time_limit=10.0):
    """
    Discrete local search with flips and swaps.
    """
    start_time = time.time()
    n = len(x)
    best_x = x.copy()
    best_c3 = _compute_c3_bound(best_x)
    
    # Precompute FFT size
    size = 1
    while size < 2 * n:
        size *= 2
    
    # Indices to consider
    indices = list(range(n))
    
    for _ in range(200):  # 200 passes
        if time.time() - start_time > time_limit:
            break
        improved = False
        np.random.shuffle(indices)
        for idx in indices:
            # Try flip
            best_x[idx] *= -1
            new_c3 = _compute_c3_bound(best_x)
            if new_c3 < best_c3:
                best_c3 = new_c3
                improved = True
            else:
                best_x[idx] *= -1
            
            # Try swap with a random other index
            if time.time() - start_time > time_limit:
                break
            other = np.random.randint(n)
            if other != idx:
                best_x[idx], best_x[other] = best_x[other], best_x[idx]
                new_c3 = _compute_c3_bound(best_x)
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    improved = True
                else:
                    best_x[idx], best_x[other] = best_x[other], best_x[idx]
        
        if not improved:
            break
    
    return best_x, best_c3

def construct():
    """
    Construct a signed step function with 400 equal intervals.
    Returns a sequence of 400 values (+1 or -1).
    """
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_seq = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tm[i] = bin(i).count('1') % 2
    tm_signed = 2 * tm - 1
    candidates.append(tm_signed)
    
    # Random sequences
    for _ in range(5):
        rand_seq = np.random.choice([-1, 1], size=N)
        candidates.append(rand_seq)
    
    # M-sequences
    def _m_sequence(k, poly):
        state = 1
        seq = []
        length = 2 ** k - 1
        for _ in range(length):
            seq.append(state & 1)
            new_bit = 0
            temp = state
            for i in range(k):
                if poly & (1 << i):
                    new_bit ^= (temp >> i) & 1
            state = (state >> 1) | (new_bit << (k - 1))
        return np.array(seq, dtype=np.int8)
    
    polys = [(9, 511), (9, 109), (10, 1023), (10, 101)]
    for k, poly in polys:
        try:
            seq = _m_sequence(k, poly)
            if len(seq) >= N:
                seq = seq[:N]
            else:
                reps = int(np.ceil(N / len(seq)))
                seq = np.tile(seq, reps)[:N]
            candidates.append(2 * seq - 1)
        except Exception:
            pass
    
    # 2. Evaluate candidates
    for seq in candidates:
        if time.time() > deadline - 30:
            break
        score = _compute_c3_bound(seq)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
    
    if best_seq is None:
        return np.ones(N)
    
    # 3. Continuous phase gradient descent on the best candidate
    if time.time() < deadline - 20:
        # Convert to complex unit-modulus
        z0 = best_seq.astype(np.complex128)
        # Initialize with small random phase perturbation
        phases = np.random.uniform(0, 2*np.pi, N)
        z0 = z0 * np.exp(1j * phases)
        
        z_opt, cost = _phase_gradient_descent(z0, max_iters=3000, lr=0.05, time_limit=15.0)
        
        # Convert back to binary
        bin_seq = _phase_to_binary(z_opt)
        # Ensure exactly +/-1
        bin_seq[bin_seq == 0] = 1
        
        # Evaluate
        bin_score = _compute_c3_bound(bin_seq)
        if bin_score < best_score:
            best_score = bin_score
            best_seq = bin_seq
    
    # 4. Discrete local search polish
    if time.time() < deadline - 5:
        opt_seq, opt_score = _discrete_local_search(best_seq, time_limit=10.0)
        if opt_score < best_score:
            best_score = opt_score
            best_seq = opt_seq
    
    return best_seq
# EVOLVE-BLOCK-END
