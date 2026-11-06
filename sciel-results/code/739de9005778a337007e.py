import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def generate_powerful(L):
        pows = set()
        # t must be squarefree
        # n = s^2 * t^3 <= L
        # t <= L^(1/3)
        max_t = int(L ** (1.0/3.0)) + 2
        
        # Primes up to sqrt(max_t) for squarefree check
        limit_p = int(max_t ** 0.5) + 1
        is_prime = [True] * limit_p
        primes = []
        for i in range(2, limit_p):
            if is_prime[i]:
                primes.append(i)
                for j in range(i*i, limit_p, i):
                    is_prime[j] = False
        
        def is_squarefree(n):
            for p in primes:
                if p * p > n:
                    break
                if n % (p * p) == 0:
                    return False
            return True

        for t in range(1, max_t + 1):
            if not is_squarefree(t):
                continue
            t3 = t ** 3
            if t3 > L:
                continue
            max_s = int((L // t3) ** 0.5)
            for s in range(1, max_s + 1):
                pows.add(s * s * t3)
        return sorted(pows)

    def run_fft_search(L):
        N_fft = 1
        # Find next power of 2 >= 2*L
        while N_fft < 2 * L:
            N_fft *= 2
            
        pows = generate_powerful(L)
        P_arr = np.array(pows, dtype=np.int64)
        
        if time.time() > deadline:
            return None
            
        # Prepare input array
        # Use float32 to save memory
        x = np.zeros(N_fft, dtype=np.float32)
        x[P_arr] = 1.0
        
        # FFT
        y = np.fft.rfft(x)
        del x
        
        if time.time() > deadline:
            return None
        
        # Square the spectrum
        y *= y
        
        # Inverse FFT
        z = np.fft.irfft(y, n=N_fft)
        del y
        
        if time.time() > deadline:
            return None
        
        # Find the best N
        # z[k] approximates the count of ordered pairs (a,b) such that a+b=k
        # We want to maximize the number of distinct pairs a <= b.
        # Let C(k) be the count of ordered pairs.
        # If k is even and k/2 is powerful, unordered count = (C(k) - 1)/2 + 1 = (C(k)+1)/2.
        # Else, unordered count = C(k)/2.
        
        P_set = set(pows)
        
        # Find candidates with high counts
        # Due to floating point errors, z values might be like 5.0000001 or 4.9999999
        max_val = np.max(z)
        # Get indices where z is close to max
        # We look for the top few to be safe
        candidates = np.where(z > max_val - 1e-5)[0]
        
        best_N = 0
        best_count = -1
        
        for N in candidates:
            N_int = int(N)
            if N_int > 2 * L:
                continue
            
            # Get the integer count
            c = int(round(z[N_int]))
            
            # Calculate unordered count
            if N_int % 2 == 0:
                half = N_int // 2
                if half in P_set:
                    count = (c + 1) // 2
                else:
                    count = c // 2
            else:
                count = c // 2
                
            if count > best_count:
                best_count = count
                best_N = N_int
        
        if best_N == 0:
            return None

        # Extract pairs
        pairs = []
        half_N = best_N // 2
        for a in P_arr:
            if a > half_N:
                break
            b = best_N - a
            if b in P_set:
                pairs.append([int(a), int(b)])
        
        return {'N': best_N, 'pairs': pairs}

    # Try large L first
    try:
        result = run_fft_search(10**9)
        if result:
            return result
    except Exception as e:
        pass # Fall through to smaller L

    # Try medium L
    try:
        result = run_fft_search(5 * 10**8)
        if result:
            return result
    except Exception as e:
        pass

    # Fallback to naive small L
    try:
        L = 10**8
        P = generate_powerful(L)
        P_set = set(P)
        counts = {}
        max_c = 0
        best_N = 0
        
        for i, a in enumerate(P):
            for b in P[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > max_c:
                    max_c = counts[s]
                    best_N = s
        
        pairs = []
        half_N = best_N // 2
        for a in P:
            if a > half_N:
                break
            b = best_N - a
            if b in P_set:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
        
    except Exception as e:
        return {'N': 2, 'pairs': [[1, 1]]}

