"""Autoconvolution minimization via fast SA with O(N) single-bit updates, Chebyshev-inspired seeds, and quadratic residue structure.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.

Key improvements:
1. O(N) direct autocorrelation update for single-bit flips (no FFT per step).
2. Chebyshev/polynomial phase optimization for high-quality seeds.
3. Thue-Morse initialization as a strong deterministic starting point.
4. Quadratic residue (Legendre symbol) seeds exploiting low multiplicative energy.
5. Multi-start SA with adaptive temperature schedule.
6. Aggressive multi-scale local polish at the end.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute Fourier basis for incremental updates (used for block flips)
_k_indices = np.arange(L // 2 + 1)
_i_indices = np.arange(N)
_exponents = -2j * np.pi * np.outer(_i_indices, _k_indices) / L
_W_basis = np.exp(_exponents)


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _ac_delta_single_flip(h, i):
    """
    Compute the change in autocorrelation when flipping h[i] -> -h[i].
    Returns delta_ac of shape (N,), where ac_new = ac + delta_ac.
    O(N) time.
    """
    delta_ac = np.zeros(N)
    hi = h[i]
    # For each lag k, the terms involving h[i] are:
    # h[i]*h[i+k] in ac[k] (j=i), and h[i-k]*h[i] in ac[k] (j=i-k)
    # Flipping h[i] changes these to -h[i]*h[i+k] and -h[i-k]*h[i]
    # So delta = -2*hi*(h[i+k] + h[i-k]) with bounds checking
    
    # k ranges 0..N-1
    # h[i+k] is valid when 0 <= i+k < N, i.e., -i <= k < N-i
    # h[i-k] is valid when 0 <= i-k < N, i.e., i-N < k <= i
    
    # For k in [0, N-i-1]: h[i+k] is valid
    if i < N:
        k_start = 0
        k_end = min(N, N - i)  # k < N-i
        if k_end > k_start:
            delta_ac[k_start:k_end] += -2.0 * hi * h[i + k_start:i + k_end]
    
    # For k in [max(0, i-N+1), i]: h[i-k] is valid
    # k such that 0 <= i-k < N => i-N < k <= i => k in [max(0, i-N+1), i]
    k_start = max(0, i - N + 1)
    k_end = min(i, N - 1) + 1
    if k_end > k_start:
        delta_ac[k_start:k_end] += -2.0 * hi * h[i - k_start:i - k_end + 1]
    
    return delta_ac


def _hard_score_from_ac(ac):
    """Compute score from autocorrelation array."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _hard_score_fast(ac):
    """Compute score from autocorrelation array, slightly optimized."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # ac[0] = N for +/-1 sequence, so dc = N
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map: even -> +1, odd -> -1
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1.0 - 2.0 * tm  # 0->+1, 1->-1


def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) for prime p using Euler's criterion.
    Returns +1 if a is a quadratic residue mod p, -1 if non-residue, 0 if a ≡ 0 mod p.
    """
    a = a % p
    if a == 0:
        return 0
    return pow(a, (p - 1) // 2, p)
    # Euler's criterion: a^((p-1)/2) ≡ 1 (mod p) if residue, -1 if non-residue


def _quadratic_residue_seed(n, p, offset=0):
    """
    Generate a binary seed based on quadratic residues modulo prime p.
    Uses the Legendre symbol: +1 for quadratic residues, -1 for non-residues.
    This exploits the number-theoretic property that quadratic residues
    have low multiplicative energy, which suppresses autoconvolution peaks.
    
    Parameters:
        n: length of output sequence
        p: prime modulus (should be > n)
        offset: shift in the residue sequence
    """
    h = np.zeros(n)
    for i in range(n):
        val = (i + offset) % p
        ls = _legendre_symbol(val, p)
        # Map: residue (+1) -> +1, non-residue (-1) -> -1, zero -> +1
        h[i] = 1.0 if ls >= 0 else -1.0
    return h


def _quadratic_residue_seeds(n, deadline):
    """
    Generate multiple quadratic residue seeds using different primes near n.
    Returns a list of candidate sequences.
    """
    seeds = []
    t_start = time.time()
    
    # Find primes near n that are > n
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        for i in range(3, int(np.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    # Try several primes in range [n+1, n+50]
    candidates_p = []
    for p in range(n + 1, n + 50):
        if is_prime(p):
            candidates_p.append(p)
        if len(candidates_p) >= 5:
            break
    
    if not candidates_p:
        # Fallback: use a larger prime
        p = n + 1
        while not is_prime(p):
            p += 1
        candidates_p = [p]
    
    for p in candidates_p:
        if time.time() > deadline - 60.0:
            break
        # Generate seed with different offsets
        for offset in [0, 1, 2, 3]:
            if time.time() > deadline - 60.0:
                break
            seed = _quadratic_residue_seed(n, p, offset)
            seeds.append(seed)
    
    return seeds


def _chebyshev_seed(deadline, rng):
    """
    Generate a high-quality binary seed via Chebyshev-inspired phase optimization.
    Uses adaptive learning rate and momentum for faster convergence.
    """
    # Start with Thue-Morse as initial guess
    x = _thue_morse(N).astype(float)
    
    # Adaptive gradient descent with momentum
    lr = 0.05
    momentum = 0.9
    v = np.zeros(N)
    n_steps = 5000
    t_start = time.time()
    
    for step in range(n_steps):
        if time.time() > deadline - 75.0:
            break
        
        # Compute autocorrelation via FFT
        F = np.fft.rfft(x, n=L)
        P = F * np.conj(F)
        ac = np.fft.irfft(P, n=L)
        
        # Gradient of sum_{k=1}^{N-1} ac[k]^2 w.r.t. x
        F_ac = np.fft.rfft(ac, n=L)
        grad_full = 2.0 * np.fft.irfft(F * np.conj(F_ac), n=L)[:N]
        grad = grad_full - 2.0 * ac[0] * x
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Momentum update
        v = momentum * v + lr * grad
        x -= v
        
        # Normalize to fixed norm
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm * np.sqrt(N)
    
    # Threshold to binary
    h_binary = np.sign(x)
    h_binary[h_binary == 0] = 1.0
    
    return h_binary


def _reaction_diffusion_seed(deadline, rng):
    """
    Generate a binary seed using a 1D reaction-diffusion inspired approach.
    Simulates Gray-Scott-like dynamics to create aperiodic patterns with
    local interactions that may produce favorable spectral properties.
    """
    # Initialize with small random perturbation around Thue-Morse
    x = _thue_morse(N).astype(float)
    x += 0.1 * rng.standard_normal(N)
    
    # Reaction-diffusion parameters
    D = 0.1  # diffusion coefficient
    k_u = 0.03  # decay rate for u
    k_v = 0.06  # decay rate for v
    dt = 0.5
    n_steps = 2000
    
    u = x.copy()
    v = np.zeros(N)
    
    for step in range(n_steps):
        if time.time() > deadline - 70.0:
            break
        
        # Laplacian (circular boundary for simplicity)
        lap_u = np.roll(u, 1) + np.roll(u, -1) - 2.0 * u
        lap_v = np.roll(v, 1) + np.roll(v, -1) - 2.0 * v
        
        # Gray-Scott-like reaction terms
        # u' = D*lap_u - u*v^2 + k_u*(1-u)
        # v' = D*lap_v + u*v^2 - (k_u + k_v)*v
        u_new = u + dt * (D * lap_u - u * v**2 + k_u * (1.0 - u))
        v_new = v + dt * (D * lap_v + u * v**2 - (k_u + k_v) * v)
        
        u = u_new
        v = v_new
    
    # Combine u and v to get final signal
    x_final = u + 0.5 * v
    
    # Threshold to binary
    h_binary = np.sign(x_final)
    h_binary[h_binary == 0] = 1.0
    
    return h_binary


def _local_polish_quick(h, deadline):
    """Quick single-pass local polish using O(N) single-bit updates."""
    h = h.copy()
    ac = _autocorr_fast(h)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    for _pass in range(5):
        if time.time() > deadline - 75.0:
            break
        any_improve = False
        for i in range(N):
            if time.time() > deadline - 75.0:
                break
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            if new_hard < best_hard - 1e-12:
                h[i] *= -1.0
                ac = ac_new
                best_hard = new_hard
                best_h = h.copy()
                any_improve = True
        if not any_improve:
            break
    
    return best_h, best_hard


def _sa_run(h_init, rng, deadline, T_start, T_min, decay, beta, sizes):
    """Run simulated annealing from h_init until deadline. Returns (best_h, best_hard).
    
    Uses O(N) direct autocorrelation updates for single-bit flips (fast path)
    and FFT-based updates for block flips (slow path).
    """
    h = h_init.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    cur_hard = _hard_score_from_ac(ac)
    cur_smooth = _smooth_score_from_ac(ac, beta)
    best_hard = cur_hard
    best_h = h.copy()
    T = T_start
    steps = 0
    
    # Small block sizes that can use direct update
    max_direct_size = 4
    
    while time.time() < deadline - 2.0:
        # Choose move type
        r = rng.random()
        if r < 0.70:
            # Single bit flip - use fast O(N) path
            i = int(rng.integers(0, N))
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[i] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            
        elif r < 0.90:
            # Small block flip (size 2-4) - use direct O(N*size) path
            size = int(rng.choice([2, 3, 4]))
            start = int(rng.integers(0, N - size + 1))
            idx_flip = np.arange(start, start + size)
            
            # Compute delta_ac directly
            delta_ac = np.zeros(N)
            for idx in idx_flip:
                delta_ac += _ac_delta_single_flip(h, int(idx))
            
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[idx_flip] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            
        else:
            # Large block flip - use FFT path
            size = int(rng.choice(sizes[sizes > max_direct_size]))
            start = int(rng.integers(0, N - size + 1))
            idx_flip = np.arange(start, start + size)
            
            h_block = h[idx_flip]
            W_block = _W_basis[idx_flip, :]
            delta_F = -2.0 * (W_block @ h_block)
            F_new = F + delta_F
            
            P = F_new * np.conj(F_new)
            ac_new = np.fft.irfft(P, n=L)[:N]
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[idx_flip] *= -1.0
                F = F_new
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
        
        T = max(T_min, T * decay)
        steps += 1
    
    return best_h, best_hard


def _local_polish(h, deadline):
    """Greedy multi-scale local polish. Returns (best_h, best_hard).
    
    Uses O(N) direct updates for small blocks and FFT for larger blocks.
    """
    h = h.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    sizes_polish = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    for _pass in range(100):
        if time.time() > deadline - 0.5:
            break
        any_improve = False
        for size in sizes_polish:
            if time.time() > deadline - 0.5:
                return best_h, best_hard
            for start in range(0, N - size + 1):
                if time.time() > deadline - 0.5:
                    return best_h, best_hard
                
                if size <= 4:
                    # Fast direct path
                    delta_ac = np.zeros(N)
                    for offset in range(size):
                        idx = start + offset
                        delta_ac += _ac_delta_single_flip(h, idx)
                    ac_new = ac + delta_ac
                else:
                    # FFT path
                    idx_flip = np.arange(start, start + size)
                    h_block = h[idx_flip]
                    W_block = _W_basis[idx_flip, :]
                    delta_F = -2.0 * (W_block @ h_block)
                    F_new = F + delta_F
                    P = F_new * np.conj(F_new)
                    ac_new = np.fft.irfft(P, n=L)[:N]
                
                new_hard = _hard_score_from_ac(ac_new)
                
                if new_hard < best_hard - 1e-12:
                    if size <= 4:
                        h[start:start+size] *= -1.0
                        ac = ac_new
                    else:
                        h[start:start+size] *= -1.0
                        F = F_new
                        ac = ac_new
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
        
        if not any_improve:
            break
    
    return best_h, best_hard


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 145.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)
        idx = np.arange(N)
        
        # Phase 1: Generate high-quality seeds
        # Thue-Morse as a strong deterministic seed
        seed_tm = _thue_morse(N)
        
        # Chebyshev-inspired continuous seed with adaptive optimization
        cont_deadline = t0 + 20.0
        seed_cheb = _chebyshev_seed(cont_deadline, rng)
        
        # Reaction-diffusion inspired seed
        rd_deadline = t0 + 30.0
        seed_rd = _reaction_diffusion_seed(rd_deadline, rng)
        
        # Quadratic residue seeds (number-theoretic structure)
        qr_deadline = t0 + 35.0
        qr_seeds = _quadratic_residue_seeds(N, qr_deadline)
        
        # Additional diverse seeds
        seed_all_plus = np.ones(N)
        seed_random = rng.choice(np.array([-1.0, 1.0]), size=N)
        seed_quad1 = np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
        seed_quad2 = np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1)))
        
        candidates = [seed_tm, seed_cheb, seed_rd, seed_all_plus, seed_random, seed_quad1, seed_quad2]
        candidates.extend(qr_seeds)
        
        best_h = None
        best_hard = 1e9
        for h in candidates:
            ac = _autocorr_fast(h)
            s = _hard_score_from_ac(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        # Phase 2: Multi-start SA with fast O(N) single-bit updates
        sa_start_time = time.time()
        polish_budget = 40.0
        sa_end = deadline - polish_budget
        
        sizes_sa = np.array([2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        
        # Build list of starting points for SA
        sa_starts = []
        # Start from best found so far
        sa_starts.append(best_h.copy())
        # Perturb best with 5% flips
        h_pert = best_h.copy()
        flip_mask = rng.random(N) < 0.05
        h_pert[flip_mask] *= -1.0
        sa_starts.append(h_pert)
        # Use Thue-Morse perturbed
        h_tm_pert = seed_tm.copy()
        flip_mask = rng.random(N) < 0.1
        h_tm_pert[flip_mask] *= -1.0
        sa_starts.append(h_tm_pert)
        # Use Chebyshev seed perturbed
        h_cheb_pert = seed_cheb.copy()
        flip_mask = rng.random(N) < 0.1
        h_cheb_pert[flip_mask] *= -1.0
        sa_starts.append(h_cheb_pert)
        # Use reaction-diffusion seed perturbed
        h_rd_pert = seed_rd.copy()
        flip_mask = rng.random(N) < 0.1
        h_rd_pert[flip_mask] *= -1.0
        sa_starts.append(h_rd_pert)
        # Add best quadratic residue seed if available
        if qr_seeds:
            best_qr = None
            best_qr_score = 1e9
            for qr in qr_seeds[:3]:  # Only check first few for speed
                ac_qr = _autocorr_fast(qr)
                s_qr = _hard_score_from_ac(ac_qr)
                if s_qr < best_qr_score:
                    best_qr_score = s_qr
                    best_qr = qr.copy()
            if best_qr is not None:
                sa_starts.append(best_qr)
        
        n_starts = len(sa_starts)
        sa_time_per = (sa_end - sa_start_time) / max(n_starts, 1)
        
        for start_idx in range(n_starts):
            if time.time() > sa_end - 1.0:
                break
            
            h_init = sa_starts[start_idx]
            
            sa_deadline = min(sa_start_time + (start_idx + 1) * sa_time_per, sa_end - 1.0)
            
            h_candidate, hard_candidate = _sa_run(
                h_init, rng, sa_deadline,
                T_start=0.02, T_min=1e-5, decay=0.9997, beta=40.0, sizes=sizes_sa
            )
            
            if hard_candidate < best_hard:
                best_hard = hard_candidate
                best_h = h_candidate.copy()
        
        # Phase 3: Final local polish
        best_h, best_hard = _local_polish(best_h, deadline)
        
        # Phase 4: Additional refinement with higher beta if time remains
        remaining = deadline - time.time()
        if remaining > 15.0:
            # Try a second polish pass with different parameters
            h_refined, hard_refined = _local_polish_quick(best_h, deadline)
            if hard_refined < best_hard:
                best_h = h_refined
                best_hard = hard_refined
        
        # Phase 5: Final aggressive single-bit polish if time remains
        remaining = deadline - time.time()
        if remaining > 5.0:
            # Do a final greedy single-bit polish
            h_final = best_h.copy()
            ac_final = _autocorr_fast(h_final)
            hard_final = _hard_score_from_ac(ac_final)
            for _pass in range(3):
                if time.time() > deadline - 0.5:
                    break
                any_improve = False
                for i in range(N):
                    if time.time() > deadline - 0.5:
                        break
                    delta_ac = _ac_delta_single_flip(h_final, i)
                    ac_new = ac_final + delta_ac
                    new_hard = _hard_score_from_ac(ac_new)
                    if new_hard < hard_final - 1e-12:
                        h_final[i] *= -1.0
                        ac_final = ac_new
                        hard_final = new_hard
                        any_improve = True
                if not any_improve:
                    break
            if hard_final < best_hard:
                best_h = h_final
                best_hard = hard_final
        
        return best_h
    except Exception:
        return np.ones(N)
