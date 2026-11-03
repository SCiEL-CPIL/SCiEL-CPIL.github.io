"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(h):
    """Compute max |autoconv| / (sum(h))^2 using FFT. Lower is better."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _compute_conv(h):
    """Compute full autoconvolution via FFT."""
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    return ac


def _m_sequence(n, k):
    """Generate an m-sequence of length n using LFSR with k stages."""
    polys = {
        4: [3, 0],
        5: [4, 3, 0],
        6: [5, 3, 0],
        7: [6, 5, 3, 0],
        8: [7, 5, 3, 2, 0],
    }
    if k not in polys:
        return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
    
    taps = polys[k]
    state = [0] * k
    state[0] = 1
    seq = []
    for _ in range(n):
        fb = 0
        for t in taps:
            fb ^= state[t]
        for i in range(k - 1, 0, -1):
            state[i] = state[i - 1]
        state[0] = fb
        seq.append(1.0 if state[0] else -1.0)
    return np.array(seq, dtype=np.float64)


def _block_pattern(n, block_size, neg_frac):
    """Generate block pattern with alternating +1/-1 blocks."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def _chunk_flip_search(current, current_score, best_score, best_h, rng, deadline, n):
    """
    Structural change: chunk-based local search with peak-targeting.
    Instead of single-bit flips, flip contiguous chunks of bits.
    Proposals are biased toward positions near autoconvolution peaks.
    """
    size = 1 << (2 * n - 1).bit_length()
    chunk_sizes = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]
    
    # Track peak location for targeting
    ac = _compute_conv(current)
    peak_idx = np.argmax(np.abs(ac))
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Choose chunk size: smaller chunks more common, larger for exploration
        r = rng.random()
        if r < 0.4:
            chunk_sz = 1
        elif r < 0.6:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.8:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32])
        
        # Peak-targeted proposal: 60% chance to target near peak, 40% uniform
        if rng.random() < 0.6:
            # Target near the autoconvolution peak
            # Peak at lag p means positions i where i or i+p or i-p are important
            # For a chunk flip at position start, it affects ac[start:end] region
            # Bias toward positions that can affect the peak
            target = peak_idx % n
            spread = max(chunk_sz, n // 8)
            start = int(rng.integers(max(0, target - spread), min(n - chunk_sz, target + spread) + 1))
        else:
            start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        
        # Ensure valid range
        start = max(0, min(start, n - chunk_sz))
        end = start + chunk_sz
        
        # Perform the chunk flip
        old_vals = current[start:end].copy()
        current[start:end] = -old_vals
        
        # Compute new score
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac_new = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
            max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        # Accept or reject
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
                # Update peak location
                peak_idx = np.argmax(np.abs(ac_new))
        else:
            # Reject and revert
            current[start:end] = old_vals
            no_improve_count += 1
    
    return current, current_score, best_score, best_h


def _multi_restart_search(best_h, best_score, rng, deadline, n):
    """
    Multiple restarts from perturbed versions of the best solution.
    Each restart does a focused chunk-flip search.
    """
    current_best = best_score
    current_best_h = best_h.copy()
    
    num_restarts = 20
    per_restart_time = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the best solution for this restart
        h = current_best_h.copy()
        # Random perturbation: flip a fraction of bits
        flip_frac = rng.choice([0.05, 0.1, 0.2, 0.3])
        n_flips = int(flip_frac * n)
        flip_indices = rng.choice(n, size=n_flips, replace=False)
        h[flip_indices] = -h[flip_indices]
        
        # Evaluate perturbed
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
        max_abs = np.max(np.abs(ac))
        score = max_abs / (s * s)
        
        # Use the better of perturbed or original as starting point
        if score < current_best:
            h = current_best_h.copy()
            score = current_best
        else:
            h = h.copy()
        
        # Run chunk-flip search from this starting point
        restart_deadline = min(time.time() + per_restart_time, deadline)
        h, score, restart_best, restart_best_h = _chunk_flip_search(
            h, score, current_best, current_best_h.copy(), rng, restart_deadline, n
        )
        
        if restart_best < current_best:
            current_best = restart_best
            current_best_h = restart_best_h
    
    return current_best_h, current_best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin from 150s limit

    rng = np.random.default_rng(42)
    
    # Diverse initializations
    candidates = []
    
    # 1. Constant ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. M-sequences with different k
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # 3. Block patterns
    for bs in [4, 8, 16, 32, 64]:
        for nf in [0.0, 0.2, 0.3, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # 4. Random with varying bias
    for bias in [0.0, 0.1, 0.2, 0.3, 0.5]:
        p_pos = 0.5 + bias / 2
        h = rng.choice([-1.0, 1.0], size=N, p=[1 - p_pos, p_pos])
        candidates.append(h)
    
    # 5. Sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        candidates.append(h)
    
    # 6. Walsh-Hadamard-like patterns
    for level in range(4):
        base = np.ones(N // (1 << level))
        # Build Kronecker-like pattern
        h = np.ones(N)
        for i in range(N):
            # XOR-based pattern
            val = 1.0
            for bit in range(level + 1):
                if (i >> bit) & 1:
                    val *= -1.0
            h[i] = val
        candidates.append(h)
    
    # Evaluate all candidates
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline:
            break
        s = _compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    n = N
    
    # Phase 1: Chunk-flip local search from the best candidate
    phase1_deadline = t0 + 60.0
    best_h, best_score, best_score, best_h = _chunk_flip_search(
        best_h.copy(), best_score, best_score, best_h.copy(), rng, phase1_deadline, n
    )
    
    # Phase 2: Multi-restart search with perturbations
    if time.time() < deadline:
        best_h, best_score = _multi_restart_search(
            best_h, best_score, rng, deadline, n
        )
    
    # Phase 3: Final refinement with chunk-flip from the best
    if time.time() < deadline:
        best_h, best_score, best_score, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, deadline, n
        )
    
    return best_h

# EVOLVE-BLOCK-END
