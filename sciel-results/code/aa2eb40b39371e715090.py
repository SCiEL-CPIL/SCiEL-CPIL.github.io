import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

def _autocorr_fast(h):
    """
    Compute the linear autocorrelation of h.
    Returns array of length N: ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
    """
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac

def _hard_score(h):
    """
    The metric to minimize: max |ac[k]| / ac[0]
    Since h is +/-1, ac[0] = N.
    So we minimize max |ac[k]|.
    """
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)

def _legendre_symbol(n, p):
    """
    Compute the Legendre symbol (n/p).
    Returns 1 if n is a quadratic residue mod p, -1 if non-residue, 0 if n%p==0.
    """
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1

def _generate_legendre_sequence(p, offset=0):
    """
    Generate a signed sequence of length N based on Legendre symbols mod p.
    h[i] = 1 if (i+offset)^2 is a quadratic residue mod p (and not 0)
    h[i] = -1 otherwise
    """
    h = np.zeros(N)
    for i in range(N):
        val = (i + offset) % p
        if val == 0:
            # 0 is not a quadratic residue in the strict sense for Legendre symbol definition usually,
            # but in sequence construction, we can map it arbitrarily. 
            # Standard Legendre sequences often exclude 0 or map it.
            # Let's map 0 to 1 for now, or just use the symbol.
            # Legendre symbol is 0 for n=0. 
            # For a balanced sequence, we might want to avoid 0 or handle it.
            # Actually, for p prime, there are (p-1)/2 residues and (p-1)/2 non-residues.
            # If we include 0, we have one extra 0.
            # Let's just use 1 for 0 to keep it binary.
            h[i] = 1.0
        else:
            sym = pow(val, (p - 1) // 2, p)
            if sym == 1:
                h[i] = 1.0
            else:
                h[i] = -1.0
    return h

def _greedy_polish(h, deadline):
    """
    Iteratively improve h by flipping single bits or small blocks if it reduces the score.
    """
    best_h = h.copy()
    best_score = _hard_score(best_h)
    
    # Precompute autocorrelation for the current best
    ac_best = _autocorr_fast(best_h)
    
    improved = True
    passes = 0
    max_passes = 1000
    
    # Block sizes to try: 1 (single bit), 2, 3, 4
    block_sizes = [1, 2, 3, 4]
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        # Try single bit flips first (cheapest to evaluate conceptually, though FFT cost is same)
        # We can optimize by noticing that flipping h[i] changes ac[k] by:
        # delta_ac[k] = -2 * h[i] * sum_{j} h[j] h[j+k] ... no, it's more complex.
        # ac_new[k] = sum (h[i']h[i'+k]) where h[i'] = -h[i'] if i'=i or i'+k=i.
        # It's safer to just recompute FFT for candidates if N is small enough, 
        # but N=400 is small. FFT is fast.
        
        # Let's iterate through positions and try flipping
        for i in range(N):
            if time.time() > deadline:
                return best_h
            
            # Try flipping bit i
            h_flip = best_h.copy()
            h_flip[i] *= -1.0
            score_flip = _hard_score(h_flip)
            
            if score_flip < best_score - 1e-12:
                best_h = h_flip
                best_score = score_flip
                improved = True
                # Continue searching from this new state immediately? 
                # Or finish the pass? 
                # Restarting the pass is often better to catch cascading improvements.
                # But for efficiency, let's just mark improved and continue.
                # Actually, if we flip, the subsequent flips in this pass are based on the old state.
                # It's better to break and restart the pass to evaluate on the new state.
                break
        
        if not improved:
            # Try block flips if single bits didn't improve
            for size in [2, 3, 4]:
                if time.time() > deadline:
                    return best_h
                for start in range(0, N - size + 1):
                    if time.time() > deadline:
                        return best_h
                    
                    h_flip = best_h.copy()
                    h_flip[start:start+size] *= -1.0
                    score_flip = _hard_score(h_flip)
                    
                    if score_flip < best_score - 1e-12:
                        best_h = h_flip
                        best_score = score_flip
                        improved = True
                        break
                if improved:
                    break
                    
    return best_h

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # Safety margin before 150s limit
        
        # 1. Generate Initial Candidates
        candidates = []
        
        # A. Random Seeds
        rng = np.random.default_rng(_SEED)
        for _ in range(5):
            candidates.append(rng.choice(np.array([-1.0, 1.0]), size=N))
            
        # B. Legendre Sequence Seeds
        # Find primes near N
        primes = [397, 401, 409, 389, 383]
        for p in primes:
            # Try different offsets to shift the sequence
            for offset in [0, 1, 2, 10, 20]:
                candidates.append(_generate_legendre_sequence(p, offset))
                
        # C. Simple Patterns (Chirps, etc.)
        idx = np.arange(N)
        candidates.append(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
        candidates.append(np.cos(2 * np.pi * idx / N))
        
        # 2. Evaluate and Select Best Initial Candidate
        best_h = None
        best_score = 1e9
        
        for h in candidates:
            # Ensure binary
            h = np.sign(h)
            h[h == 0] = 1.0
            score = _hard_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
        # 3. Polish with Greedy Local Search
        if time.time() < deadline - 5.0:
            best_h = _greedy_polish(best_h, deadline)
            
        # 4. Final Check
        # It's possible that a different random restart + polish could be better, 
        # but given the time constraints, one strong polish on the best seed is usually sufficient.
        # If time remains, try one more random restart.
        if time.time() < deadline - 10.0:
            h_rand = rng.choice(np.array([-1.0, 1.0]), size=N)
            h_polished = _greedy_polish(h_rand, deadline - 2.0)
            score_polished = _hard_score(h_polished)
            if score_polished < best_score:
                best_h = h_polished
                best_score = score_polished

        return best_h

    except Exception:
        return np.ones(N)
