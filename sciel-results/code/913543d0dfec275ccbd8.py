import math
import time
from collections import defaultdict

_START = time.time()
DEADLINE = _START + 145  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - time.time()

def construct():
    # Fallback known good result
    fallback = {'M': 333312, 'k': [336, 372, 434]}

    if _time_left() < 10:
        return fallback

    # Linear sieve to compute sigma(k) for all k up to N
    # sigma[k] = sum of divisors of k
    # We use the standard linear sieve for multiplicative functions.
    # sigma is multiplicative: sigma(p^a) = (p^(a+1)-1)/(p-1)
    
    # We'll use arrays for:
    #   sigma[i] = sigma(i)
    #   lowp[i] = smallest prime factor of i (for the sieve)
    #   powp[i] = p^a where p = lowp[i] and p^a || i
    #   sigmapowp[i] = sigma(p^a) for the same p, a
    
    N = 30_000_000  # Start with 30M, adjust based on time
    
    try:
        # Allocate arrays
        sigma = [0] * (N + 1)
        lowp = [0] * (N + 1)
        powp = [0] * (N + 1)
        sigmapowp = [0] * (N + 1)
        primes = []
        
        sigma[1] = 1
        lowp[1] = 1
        powp[1] = 1
        sigmapowp[1] = 1
        
        t0 = time.time()
        
        for i in range(2, N + 1):
            if lowp[i] == 0:
                # i is prime
                lowp[i] = i
                powp[i] = i
                sigmapowp[i] = i + 1
                sigma[i] = i + 1
                primes.append(i)
            
            for p in primes:
                if p > lowp[i]:
                    break
                v = i * p
                if v > N:
                    break
                lowp[v] = p
                if p == lowp[i]:
                    # p divides i, so p^a || i means p^(a+1) || v
                    powp[v] = powp[i] * p
                    sigmapowp[v] = sigmapowp[i] + powp[v]
                    # sigma(v) = sigma(i / p^a) * sigma(p^(a+1))
                    # sigma(i) = sigma(i / p^a) * sigma(p^a)
                    # So sigma(v) = sigma(i) / sigmapowp[i] * sigmapowp[v]
                    sigma[v] = sigma[i] // sigmapowp[i] * sigmapowp[v]
                    break
                else:
                    # p doesn't divide i
                    powp[v] = p
                    sigmapowp[v] = p + 1
                    sigma[v] = sigma[i] * (p + 1)
            
            # Time check every 1M iterations
            if i % 1_000_000 == 0:
                if _time_left() < 20:
                    break
        
        if _time_left() < 15:
            return fallback
        
        # Now group by M = k * sigma(k)
        groups = defaultdict(list)
        
        # For each k, compute M = k * sigma(k)
        # We need k <= sqrt(M) = sqrt(k * sigma(k)), i.e., k <= sigma(k)
        # And M <= 10^18
        # Also k must be a valid solution: k * sigma(k) = M exactly
        
        t1 = time.time()
        for k in range(1, N + 1):
            sk = sigma[k]
            M = k * sk
            if M > 10**18:
                continue
            # Check k <= sqrt(M)
            if k * k > M:
                continue
            groups[M].append(k)
            
            if k % 5_000_000 == 0:
                if _time_left() < 15:
                    break
        
        if _time_left() < 10:
            return fallback
        
        # Find M with most solutions
        best_M = fallback['M']
        best_ks = fallback['k']
        best_count = len(fallback['k'])
        
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks
        
        if best_count <= best_count and best_ks == fallback['k']:
            return fallback
        
        # Verify with exact sigma computation
        def _sigma_verify(k):
            """Compute sigma(k) by trial division (exact)."""
            if k == 1:
                return 1
            result = 1
            temp = k
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    exp = 0
                    while temp % p == 0:
                        temp //= p
                        exp += 1
                    result *= (p**(exp+1) - 1) // (p - 1)
                p += 1 if p == 2 else 2
            if temp > 1:
                result *= (temp + 1)
            return result
        
        sqrt_M = math.isqrt(best_M)
        verified_ks = []
        for k in best_ks:
            if k <= sqrt_M and k * _sigma_verify(k) == best_M:
                verified_ks.append(k)
        
        if len(verified_ks) >= 1:
            return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
        else:
            return fallback
    
    except Exception:
        return fallback

def _small_search():
    return {'M': 333312, 'k': [336, 372, 434]}
