"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    def compute_autoconv(h):
        """Compute autoconvolution lags 0..N-1"""
        # Use FFT for initial computation
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        ac = np.fft.irfft(H * np.conj(H), n=n_fft)
        # ac[k] for k=0..N-1 gives autoconv[k]
        return ac[:N].astype(np.int64)
    
    def compute_score(autoconv, S):
        """Score = max|autoconv| / S^2"""
        max_ac = np.max(np.abs(autoconv))
        if S == 0:
            return float('inf')
        return max_ac / (S * S)
    
    def try_flip(h, p, autoconv, S):
        """Try flipping position p, return new score if improved, else None"""
        old_hp = h[p]
        # Compute change in autoconv for each lag k
        # For lag k: change = -2 * h[p] * (h[p-k] if 0<=p-k<N else 0 + h[p+k] if 0<=p+k<N else 0)
        # We need to check all k from 0 to N-1
        
        # For efficiency, let's compute the new autoconv incrementally
        # new_autoconv[k] = autoconv[k] - 2*h[p]*(h[p-k] + h[p+k]) for valid indices
        
        # Let's compute the effect on max|autoconv| and S
        new_S = S - 2 * old_hp
        
        # For each lag k, the change is:
        # delta_k = -2 * old_hp * (h[(p-k) % N] if we're doing circular... no, it's linear)
        # Linear convolution: h[j] = 0 for j < 0 or j >= N
        
        # For lag k (0 <= k < N):
        # Terms: h[p]*h[p-k] (if 0 <= p-k < N) and h[p+k]*h[p] (if 0 <= p+k < N)
        # So delta_k = -2 * old_hp * (h[p-k] if 0<=p-k<N else 0 + h[p+k] if 0<=p+k<N else 0)
        
        # We need to find the new max|autoconv| efficiently.
        # This is tricky. Let's just compute the new autoconv for the affected lags.
        # The affected lags are those k where p-k is in [0,N-1] or p+k is in [0,N-1]
        # That's k in [max(0, p-(N-1)), min(N-1, p+(N-1))] which is essentially all k for most p.
        
        # For N=400, let's just recompute the relevant parts.
        # Actually, let's compute the delta for all k and update.
        
        # Create delta array
        delta = np.zeros(N, dtype=np.int64)
        for k in range(N):
            term = 0
            if 0 <= p - k < N:
                term += h[p - k]
            if 0 <= p + k < N:
                term += h[p + k]
            delta[k] = -2 * old_hp * term
        
        new_autoconv = autoconv + delta
        
        new_max_ac = np.max(np.abs(new_autoconv))
        if new_S == 0:
            return None, None, None
        new_score = new_max_ac / (new_S * new_S)
        
        return new_score, new_autoconv, new_S
    
    # Multiple random restarts
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random signed sequence
        h = np.random.choice([-1, 1], size=N)
        
        # Compute initial autoconv and S
        autoconv = compute_autoconv(h)
        S = int(np.sum(h))
        score = compute_score(autoconv, S)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Local search: try flipping each position
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each position in random order
            order = np.random.permutation(N)
            for p in order:
                if time.time() > deadline:
                    break
                
                new_score, new_autoconv, new_S = try_flip(h, int(p), autoconv, S)
                
                if new_score is not None and new_score < score:
                    # Accept the flip
                    h[p] = -h[p]
                    autoconv = new_autoconv
                    S = new_S
                    score = new_score
                    improved = True
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
