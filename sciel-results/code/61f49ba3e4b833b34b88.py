import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s
    
    def elapsed():
        return time.time() - start_time
    
    # Generate powerful numbers up to L
    def gen_powerful(L):
        # b is squarefree, b^3 <= L
        b_max = int(round(L ** (1/3))) + 1
        # Generate squarefree numbers up to b_max
        # A number is squarefree iff it's not divisible by p^2 for any prime p
        # Use sieve-like approach
        is_sqfree = np.ones(b_max + 1, dtype=bool)
        is_sqfree[0] = False
        # Sieve for primes
        is_prime = np.ones(b_max + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(np.sqrt(b_max)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        # Mark non-squarefree
        for i in range(2, int(np.sqrt(b_max)) + 1):
            if is_prime[i]:
                is_sqfree[i*i::i*i] = False
        squarefree = [b for b in range(1, b_max + 1) if is_sqfree[b]]
        
        powerful_set = set()
        for b in squarefree:
            b3 = b ** 3
            a_max = int(np.sqrt(L // b3))
            for a in range(1, a_max + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        return sorted(powerful_set)
    
    # Try different L values, starting large
    L_candidates = [10**8, 5*10**7, 2*10**7, 10**7, 5*10**6, 2*10**6]
    
    best_result = None
    best_score = 0
    
    for L in L_candidates:
        if elapsed() > deadline - 30:  # need 30s for FFT and extraction
            break
        
        powerful = gen_powerful(L)
        if elapsed() > deadline - 30:
            break
        
        n_pow = len(powerful)
        # Create indicator array
        f = np.zeros(2 * L, dtype=np.float64)
        f[powerful] = 1.0
        
        # FFT convolution
        # Need FFT size >= 2L for linear convolution
        fft_size = 1
        while fft_size < 2 * L:
            fft_size *= 2
        # Actually, for convolution of two arrays of size 2L, we need FFT size >= 4L
        # Wait, f has length 2L. The convolution f*f has length 4L-1.
        # So FFT size >= 4L.
        # But we only care about indices up to L (since a,b <= L, a+b <= 2L, but we want N <= L)
        # Actually a and b are powerful numbers <= L, so a+b <= 2L.
        # We want N up to L, so we need the convolution up to index L.
        # FFT size >= 2L should suffice for indices up to L.
        
        # Let me reconsider: f has non-zero entries at indices in [1, L].
        # (f*f)[n] = sum_k f[k] * f[n-k]
        # For n <= L, k ranges over [1, n-1] roughly, all within [1, L].
        # So we need the convolution computed correctly for indices 0 to L.
        # Linear convolution of two length-2L arrays needs FFT size >= 4L-1.
        # But since f is zero for indices > L, effectively the arrays have length L+1.
        # Convolution of two length-(L+1) arrays needs FFT size >= 2(L+1)-1 = 2L+1.
        # So FFT size >= 2L+1, round up to next power of 2.
        
        fft_size = 1
        while fft_size < 2 * L + 1:
            fft_size *= 2
        
        if fft_size > 2**28:  # too large
            continue
        
        F = np.fft.rfft(f, n=fft_size)
        G = np.fft.irfft(F * F, n=fft_size)
        
        # G[n] = number of ordered pairs (a,b) with a+b=n, both powerful
        # For n from 2 to L:
        # distinct pairs with a<=b: (G[n] + I(n even and n/2 powerful)) / 2
        
        # Check time
        if elapsed() > deadline - 10:
            break
        
        # Compute scores for all N
        # G is real (from irfft), but may have tiny imaginary parts... no, irfft returns real.
        G = G[:L+1]  # only need up to L
        
        # I(n even and n/2 powerful)
        # Create array for this
        half_powerful = np.zeros(L+1, dtype=np.float64)
        for p in powerful:
            if p * 2 <= L:
                half_powerful[p * 2] = 1.0
        
        # scores[n] = (G[n] + half_powerful[n]) / 2 for n >= 2
        # But G[n] should be integer (count of ordered pairs)
        # Due to FFT numerical error, round to nearest integer
        G_rounded = np.round(G).astype(np.float64)
        
        scores = np.zeros(L+1, dtype=np.float64)
        scores[2:] = (G_rounded[2:] + half_powerful[2:]) / 2.0
        
        max_idx = int(np.argmax(scores[2:])) + 2
        max_score = scores[max_idx]
        
        if max_score > best_score:
            best_score = max_score
            best_N = max_idx
            best_L = L
            best_G = G_rounded
            best_half = half_powerful
            best_powerful = powerful
            best_result = max_idx
        
        if elapsed() > deadline - 15:
            break
    
    if best_result is None:
        # Fallback: small brute force
        L = 100000
        powerful = gen_powerful(L)
        pset = set(powerful)
        best_n = 2
        best_pairs = []
        for n in range(2, L+1):
            pairs = []
            for a in powerful:
                if 2*a > n:
                    break
                b = n - a
                if b in pset:
                    pairs.append([a, b])
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_n = n
        return {'N': best_n, 'pairs': best_pairs}
    
    # Extract pairs for best_N
    N = best_N
    pairs = []
    pset = set(best_powerful)
    for a in best_powerful:
        if 2 * a > N:
            break
        b = N - a
        if b in pset:
            pairs.append([a, b])
    
    return {'N': N, 'pairs': pairs}
