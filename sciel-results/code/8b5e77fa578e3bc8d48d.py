"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 50.0  # safety margin vs 90s kill
_BIG = 10**9

# Precompute the incidence matrix for all arithmetic progressions
# M[k, i] = 1 if index i is in the k-th AP, else 0
# The sum of signs for AP k is dot(M[k], s)
_M = None

def _build_matrix():
    """Build the dense binary incidence matrix for all APs in [0, N-1]."""
    global _M
    n = _N
    # Estimate number of APs to pre-allocate? 
    # Or just use a list of rows and stack them.
    # List of rows might be memory heavy if we append numpy arrays.
    # Better to estimate size or use a large buffer.
    # Let's just append to a list of arrays, then np.vstack.
    # Or even better, np.zeros with a generous estimate.
    
    # Estimate: 
    # d=1: ~33k
    # d=2: ~16k
    # d=3: ~11k
    # d=4: ~8k
    # Sum decays. Total ~ 100k - 150k.
    # Let's allocate 200,000 rows to be safe.
    max_rows = 200_000
    M = np.zeros((max_rows, n), dtype=np.float32)
    row_idx = 0
    
    for d in range(1, n + 1):
        for r in range(d):
            # Get indices for this residue class
            idx = list(range(r, n, d))
            L = len(idx)
            if L == 0:
                continue
            
            # Generate all contiguous sub-sequences of idx
            # For start i in 0..L-1:
            #   For end j in i..L-1:
            #     The AP is idx[i], ..., idx[j]
            #     Set M[row_idx, idx[i:j+1]] = 1
            for i in range(L):
                for j in range(i, L):
                    if row_idx >= max_rows:
                        # Expand if necessary (unlikely)
                        # This is a safety check, should not trigger
                        pass 
                    # Vectorized set
                    M[row_idx, idx[i:j+1]] = 1.0
                    row_idx += 1
                    
    # Trim to actual size
    _M = M[:row_idx]
    return _M

def _max_ap_abs(s):
    """Return the max absolute signed sum over ANY arithmetic progression."""
    # s is a numpy array of shape (N,)
    sums = _M @ s
    return float(np.max(np.abs(sums)))

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    n = _N
    
    # Build matrix once
    t0 = time.time()
    _build_matrix()
    t1 = time.time()
    # print(f"Matrix build time: {t1-t0:.2f}s") # Debug
    
    best = None
    best_w = _BIG
    
    # Multi-restart Simulated Annealing
    restart = 0
    while time.time() < deadline - 2.0:
        # Initialize
        if restart == 0:
            # Start with alternating pattern
            s = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n)], dtype=np.float32)
        else:
            s = np.array([rng.choice([-1.0, 1.0]) for _ in range(n)], dtype=np.float32)
        
        cur_w = _max_ap_abs(s)
        if cur_w < best_w:
            best_w = cur_w
            best = s.copy()
            
        # Simulated Annealing
        T = 1.0
        local_deadline = time.time() + 3.0
        if local_deadline > deadline - 2.0:
            local_deadline = deadline - 2.0
            
        while time.time() < local_deadline:
            idx = rng.randrange(n)
            # Flip sign
            old_val = s[idx]
            s[idx] = -old_val
            
            new_w = _max_ap_abs(s)
            
            # Acceptance criterion
            if new_w <= cur_w or rng.random() < np.exp(-(new_w - cur_w) / T):
                cur_w = new_w
                if new_w < best_w:
                    best_w = new_w
                    best = s.copy()
                    if best_w == 1.0:
                        break
            else:
                s[idx] = old_val
            
            # Cool down
            T = max(0.01, T * 0.999)
            
        restart += 1
        
    # Final safety
    if best is None:
        best = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n)], dtype=np.float32)
        
    # Convert to Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
