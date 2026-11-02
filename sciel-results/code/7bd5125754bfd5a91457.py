import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _random_bitseq(rng):
    x = rng.random(N)
    idx = np.argsort(x)
    seq = np.zeros(N)
    seq[idx[HALF:]] = 1.0
    return seq

def _sa(rng, start, iters, T0, T1, t_start, deadline):
    h = start.copy()
    cur = _max_overlap(h)
    best = h.copy()
    bestc = cur
    for it in range(iters):
        if time.time() > deadline:
            break
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        h[i], h[j] = h[j], h[i]
        newc = _max_overlap(h)
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        if newc < cur or rng.random() < np.exp(-(newc - cur) / max(T, 1e-9)):
            cur = newc
            if newc < bestc:
                bestc = newc
                best = h.copy()
        else:
            h[i], h[j] = h[j], h[i]
    return best, bestc

def _local_search(h, t_start, deadline):
    """
    Exhaustive single-swap local search.
    Tries every possible swap of a 1 and a 0.
    """
    h = h.copy()
    cur = _max_overlap(h)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        for i in ones:
            for j in zeros:
                if time.time() > deadline:
                    return h, cur
                orig_i, orig_j = h[i], h[j]
                h[i], h[j] = h[j], h[i]
                newc = _max_overlap(h)
                if newc < cur:
                    cur = newc
                    improved = True
                    break
                else:
                    h[i], h[j] = orig_i, orig_j
            if improved:
                break
    return h, cur

def _annealed_ls(rng, h_init, t_start, deadline, n_restarts=50):
    """
    Hybrid: Run SA for a short time, then LS.
    Repeat with random perturbations.
    """
    best = h_init.copy()
    bestc = _max_overlap(h_init)
    
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb current best
        h = best.copy()
        # Randomly swap 5-10 pairs
        n_swaps = rng.integers(5, 11)
        for _ in range(n_swaps):
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h[i], h[j] = h[j], h[i]
        
        # Short SA
        h_sa, c_sa = _sa(rng, h, iters=2000, T0=0.5, T1=0.01, t_start=t_start, deadline=deadline)
        
        # LS
        h_ls, c_ls = _local_search(h_sa, t_start, deadline)
        
        if c_ls < bestc:
            bestc = c_ls
            best = h_ls
            
    return best, bestc

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    bestc = np.inf
    deadline = t0 + DEADLINE

    # Strategy:
    # 1. Generate diverse binary starting points.
    # 2. Run SA on each.
    # 3. Refine with LS.
    # 4. Use "Annealed LS" to escape local minima.

    starts = []
    # Alternating
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(alt)
    # Block
    blk = np.zeros(N); blk[:HALF] = 1.0
    starts.append(blk)
    # Random
    for _ in range(8):
        starts.append(_random_bitseq(rng))

    # Phase 1: SA + LS on diverse starts
    # Allocate ~50% of time to initial exploration
    phase1_deadline = t0 + (DEADLINE * 0.5)
    
    for s in starts:
        if time.time() > phase1_deadline:
            break
        h_sa, c_sa = _sa(rng, s, iters=10000, T0=2.0, T1=0.05, t_start=t0, deadline=phase1_deadline)
        h_ls, c_ls = _local_search(h_sa, t0, phase1_deadline)
        if c_ls < bestc:
            bestc = c_ls
            best = h_ls

    if best is None:
        best = np.full(N, 0.5)
        bestc = _max_overlap(best)

    # Phase 2: Annealed LS to escape local minima
    # Use remaining time
    phase2_deadline = deadline - 1.0
    best, bestc = _annealed_ls(rng, best, t0, phase2_deadline, n_restarts=100)

    # Final check: ensure sum is exactly HALF
    # Binary vectors from SA/LS always sum to HALF.
    
    return best
