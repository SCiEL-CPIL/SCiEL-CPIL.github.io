"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        size = 1
        while size < 2 * len(x) - 1:
            size *= 2
        F = np.fft.rfft(x, size)
        ac = np.fft.irfft(F * np.conj(F), size)[:len(x)]
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    rng = np.random.default_rng(42)

    def make_mseq(m, poly_coeff):
        L = 2**m - 1
        seq = [0] * L
        state = 1
        for i in range(L):
            seq[i] = 1 if state & 1 else -1
            fb = 0
            for exp in poly_coeff:
                if exp < m:
                    fb ^= (state >> exp) & 1
            state = ((state >> 1) | (fb << (m-1))) & ((1 << m) - 1)
        return np.array(seq)

    candidates = []
    
    # 1. Random
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. M-sequences
    polys = [
        [9, 4, 0],
        [9, 5, 0],
        [9, 6, 0],
        [10, 7, 0],
        [10, 8, 0],
    ]
    for p in polys:
        m = p[0]
        seq = make_mseq(m, p)
        if len(seq) >= N:
            candidates.append(seq[:N].copy())
            
    # 3. Gold Codes
    s1 = make_mseq(9, [9, 4, 0])
    s2 = make_mseq(9, [9, 5, 0])
    gold = np.sign(s1[:N] + s2[:N])
    gold[gold == 0] = 1.0
    candidates.append(gold)

    # 4. Legendre sequence
    p = 401
    leg = np.ones(p)
    for i in range(1, p):
        if pow(i, (p-1)//2, p) == p-1:
            leg[i] = -1
    candidates.append(leg[:N].copy())

    best_x = np.zeros(N)
    best_val = float('inf')
    
    for c in candidates:
        s = np.sum(c)
        if abs(s) < N * 0.1:
            c = c * np.sign(s) if s != 0 else c
            c = np.sign(c + 0.1)
            c[c == 0] = 1.0
            
        val = score(c)
        if val < best_val:
            best_val = val
            best_x = c.copy()
            
        x = c.copy()
        val = score(x)
        
        improved = True
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            if not improved:
                break
            improved = False
            
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            orig1, orig2 = x[idx1], x[idx2]
            
            x[idx1] = -orig1
            x[idx2] = -orig2
            
            new_val = score(x)
            if new_val < val - 1e-12:
                val = new_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                improved = True
            else:
                x[idx1] = orig1
                x[idx2] = orig2

    return best_x
# EVOLVE-BLOCK-END
