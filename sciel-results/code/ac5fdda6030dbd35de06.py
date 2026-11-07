import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145  # safety margin from 150s limit

    # Initial best (known valid solution)
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    # Strategy:
    # 1. Fast sieve for small k (up to ~5M) using numpy to find collisions quickly.
    # 2. Targeted search for "smooth" k values (numbers with many small prime factors)
    #    because these tend to have larger sigma(k)/k ratios, leading to more
    #    structured collisions or simply finding more solutions for specific M.
    #    Actually, the key insight from inspirations is that direct enumeration
    #    of a dense range finds collisions. The previous attempt failed due to
    #    Python loop overhead in the sieve. We will use vectorized numpy operations
    #    where possible or a highly optimized sieve.
    
    # Let's try a segmented sieve or just a very optimized linear sieve in Python
    # with numpy for the sigma calculation part?
    # Actually, computing sigma(k) for all k up to N is O(N log N) if done naively.
    # A linear sieve for sigma is O(N).
    
    # Let's implement a fast linear sieve for sigma using Python lists, but optimized.
    # Or better, use the divisor sum property:
    # sigma(n) = sum_{d|n} d
    # We can compute M_k = k * sigma(k) for k=1..N.
    
    # Alternative approach:
    # Instead of sieving k, let's look for M that are "highly composite" in terms of
    # the number of solutions.
    # The condition is k * sigma(k) = M.
    # For small k, M is small.
    # Let's try to extend the sieve to N=10M using a C-like speed via PyPy? No, we are in CPython.
    # We need to be very efficient.
    
    # Let's try a different structural change:
    # Generate M values from small prime powers and combinations.
    # If M = p^a, then k must be p^b? No.
    
    # Let's stick to the sieve but make it as fast as possible.
    # We can use `array` module or `numpy` for the SPF array.
    # The bottleneck in the previous code was the Python loop for the linear sieve.
    
    # Let's try a simple O(N log N) sieve for sigma which is easier to vectorize with numpy.
    # sigma = np.ones(N+1)
    # for i in range(1, N+1):
    #     for j in range(i, N+1, i):
    #         sigma[j] += i
    # This is too slow in Python loops.
    
    # But we can do it with numpy slicing!
    # sigma = np.ones(N+1, dtype=np.int64)
    # for i in range(1, N+1):
    #     sigma[i::i] += i
    # This is still a Python loop over N, but the inner update is vectorized.
    # For N=5M, 5M iterations of numpy slice updates might be feasible?
    # Each update is O(N/i). Total operations N log N ~ 5M * 15 = 75M ops.
    # Numpy overhead per slice is small. This might work for N=2M-5M.
    
    def fast_sigma_sieve(N):
        """
        Compute M_k = k * sigma(k) for k=1..N using numpy vectorized divisor sum.
        Returns a dictionary M -> list of k.
        """
        import numpy as np
        
        # Use int64 for sigma and M
        sigma = np.ones(N + 1, dtype=np.int64)
        
        # Vectorized divisor sum
        # This loop is the bottleneck.
        # To speed it up, we can process in chunks? No, the dependency is on i.
        # Let's just try it.
        
        # Optimization: only iterate up to N
        # We can skip even i? No, sigma depends on all divisors.
        
        # Let's try to limit N to a value that completes in time.
        # Estimate: 5M iterations. If each takes 10us, that's 50s. Too slow.
        # If each takes 1us, that's 5s.
        # Numpy slice assignment is quite fast.
        
        # Let's try N=1M first to see how long it takes.
        # If 1M takes T seconds, we can scale.
        
        # Actually, there is a better way.
        # Use the linear sieve for sigma in Python, but optimized with local variables.
        
        # Let's try the linear sieve approach again but with minimal overhead.
        
        spf = [0] * (N + 1)
        primes = []
        sigma = [0] * (N + 1)
        pp = [0] * (N + 1)      # highest power of spf[k] dividing k
        sig_pp = [0] * (N + 1)  # sigma(pp[k])
        rest = [0] * (N + 1)    # k // pp[k]
        
        sigma[1] = 1
        pp[1] = 1
        sig_pp[1] = 1
        rest[1] = 1
        
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                pp[i] = i
                sig_pp[i] = 1 + i
                rest[i] = 1
                sigma[i] = 1 + i
            
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
                
                if p == spf[i]:
                    # p divides i
                    pp[i * p] = pp[i] * p
                    sig_pp[i * p] = sig_pp[i] + pp[i * p]
                    rest[i * p] = rest[i]
                else:
                    # p does not divide i
                    pp[i * p] = p
                    sig_pp[i * p] = 1 + p
                    rest[i * p] = i
                
                if rest[i * p] > 1:
                    sigma[i * p] = sig_pp[i * p] * sigma[rest[i * p]]
                else:
                    sigma[i * p] = sig_pp[i * p]
            
            # Check time
            if i % 100000 == 0:
                if time.time() > deadline:
                    break
        
        # Now collect collisions
        m_to_k = defaultdict(list)
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                m_to_k[M].append(k)
        
        return m_to_k

    # Try increasing N
    # We will try N=1M, 2M, 5M, 10M
    # But we must be careful with memory and time.
    # The linear sieve is O(N) but the constant factor in Python is high.
    # 10M iterations in Python loop might take 5-10 seconds.
    # The inner loop for primes is short on average.
    
    target_ns = [1_000_000, 2_000_000, 5_000_000, 10_000_000]
    
    for N in target_ns:
        if time.time() > deadline - 5:  # leave 5s for cleanup
            break
        
        try:
            m_to_k = fast_sigma_sieve(N)
            
            # Find the M with the most solutions
            for M, k_list in m_to_k.items():
                if len(k_list) > best_count:
                    best_count = len(k_list)
                    best_M = M
                    best_k_list = list(k_list)
                    # Sort for consistency
                    best_k_list.sort()
                    
        except MemoryError:
            break
        except Exception as e:
            # print(e)
            break

    # If the sieve didn't find anything better, try a targeted search for larger k?
    # Given the time constraints, the sieve up to 5M-10M is the most promising structural improvement
    # over the previous buggy/slow implementation.
    
    # Ensure types are correct
    best_k_list = [int(x) for x in best_k_list]
    
    return {'M': int(best_M), 'k': best_k_list}
