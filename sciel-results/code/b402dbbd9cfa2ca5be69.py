import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Sieve of Eratosthenes to find primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_sieve(limit):
        """
        Compute phi(n) for all n in [1, limit] using a linear-like sieve approach.
        Returns a list phi[0..limit] where phi[n] is Euler's totient of n.
        phi[0] = 0.
        """
        phi = list(range(limit + 1))
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi[j] -= phi[j] // i
        return phi

    def find_best_run(phi_vals, offset=0):
        """
        Find the longest run of consecutive distinct phi values.
        phi_vals is a list where phi_vals[i] corresponds to number (offset + i).
        Returns (max_length, best_start_number).
        """
        best_k = 0
        best_start = offset
        seen = set()
        run_start = 0
        run_len = 0

        for i, v in enumerate(phi_vals):
            if v not in seen:
                seen.add(v)
                run_len += 1
            else:
                if run_len > best_k:
                    best_k = run_len
                    best_start = offset + run_start
                # Reset seen, but keep the current value as the start of the new run
                # Note: If v was in seen, it means v == phi(offset + i - j) for some j.
                # The new run starts at i.
                seen = {v}
                run_start = i
                run_len = 1

        if run_len > best_k:
            best_k = run_len
            best_start = offset + run_start

        return best_k, best_start

    def phi_trial(n, primes):
        """Compute phi(n) using trial division by primes."""
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def check_window_large(start, length, primes):
        """
        Check a window of consecutive integers starting at 'start' for distinct phi values.
        Returns the length of the longest distinct prefix.
        """
        seen = set()
        k = 0
        for i in range(length):
            n = start + i
            if n <= 0:
                continue
            v = phi_trial(n, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Precompute primes for trial division up to sqrt(10^12) = 10^6
    # This takes a bit of time, so do it early.
    primes_for_trial = sieve_primes(10**6)

    best_k = 1
    best_start = 1

    # Phase 1: Sieve Search
    # Start with a safe limit
    sieve_limits = [10**6, 5 * 10**6, 2 * 10**7, 5 * 10**7]
    
    for limit in sieve_limits:
        if time_remaining() < 10.0:
            break
        
        try:
            phi_vals = compute_phi_sieve(limit)
            # phi_vals[0] is 0, we care about n >= 1
            # find_best_run expects values for n=offset+i
            # Here offset=1, so phi_vals[1] corresponds to n=1.
            # We pass phi_vals[1:] and offset=1.
            k, s = find_best_run(phi_vals[1:], offset=1)
            if k > best_k:
                best_k = k
                best_start = s
            
            # If we found a very long run, we might not need to search larger sieves
            # but let's continue if time permits as larger ranges might have longer runs.
            
        except MemoryError:
            break
        except Exception:
            continue

    # Phase 2: Check extension of the best found run if it was near the sieve limit
    # If best_start + best_k - 1 was close to the last sieve limit, try to extend it using trial division
    if time_remaining() > 5.0:
        # We need to know the last successful sieve limit
        # Let's just re-evaluate the best candidate from the sieve with a larger window using trial division
        # to see if it extends.
        # Actually, find_best_run gave us the max run *within* the sieve.
        # If the run ended at the sieve limit, it might continue.
        # Let's check the window [best_start, best_start + best_k + 20] using trial division.
        # This is safe because we already know the first best_k are distinct.
        
        # But we should also search for new runs in the large range.
        
        # Phase 2.1: Random Search in Large Range
        random.seed(42)
        
        # Number of samples depends on time
        max_samples = 500
        samples_done = 0
        
        # We want to find runs longer than best_k
        # So we check windows of size best_k + 10
        check_len = best_k + 20
        
        while time_remaining() > 2.0 and samples_done < max_samples:
            # Sample start from [10^7, 10^12]
            # Bias towards larger numbers? No, uniform is fine.
            # Actually, phi(n) grows roughly linearly, so collisions are roughly uniform in probability?
            # Let's just sample uniformly.
            
            # Avoid overlapping with the sieve range too much, though it's okay.
            low_bound = 10**7
            high_bound = 10**12
            
            s = random.randint(low_bound, high_bound - check_len)
            
            # Check this window
            k_found = check_window_large(s, check_len, primes_for_trial)
            
            if k_found > best_k:
                best_k = k_found
                best_start = s
                # If we found a longer run, we can update check_len to try to extend it further?
                # Or just keep checking with the new best_k + 20.
                check_len = best_k + 20
            
            samples_done += 1
            
            # Occasional check for time
            if samples_done % 50 == 0:
                if time_remaining() < 5.0:
                    break

    # Phase 3: Local Refinement
    # Try to extend the best found run slightly
    if time_remaining() > 1.0 and best_k > 1:
        # Try shifting the start of the best run by small amounts
        # This might catch a run that starts 1 or 2 numbers earlier/later and is longer
        for delta in range(-5, 6):
            if time_remaining() < 0.5:
                break
            s_try = best_start + delta
            if s_try < 1 or s_try + best_k + 10 > 10**12:
                continue
            
            # Check a window around s_try
            # We know best_k is the current best. Let's check for best_k + 5
            k_try = check_window_large(s_try, best_k + 10, primes_for_trial)
            if k_try > best_k:
                best_k = k_try
                best_start = s_try

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
