import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def compute_phi_range(start, length):
        """
        Computes phi(n) for n in [start, start+length-1].
        """
        end = start + length - 1
        phi = list(range(start, end + 1))
        
        for p in primes:
            if p * p > end:
                break
            first_multiple = ((start + p - 1) // p) * p
            for j in range(first_multiple, end + 1, p):
                if phi[j - start] % p == 0:
                    phi[j - start] = phi[j - start] // p * (p - 1)
        
        for i in range(length):
            if phi[i] > 1:
                phi[i] = phi[i] // phi[i] * (phi[i] - 1)
                
        return phi

    def find_max_run_in_phi_list(phi_list, max_k=200):
        """
        Find the longest run of consecutive distinct values in phi_list.
        """
        n = len(phi_list)
        if n == 0:
            return 0, 0
        
        best_len = 0
        best_start = 0
        counts = {}
        left = 0
        
        for right in range(n):
            val = phi_list[right]
            counts[val] = counts.get(val, 0) + 1
            
            while counts[val] > 1:
                left_val = phi_list[left]
                counts[left_val] -= 1
                left += 1
                
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = left
                
            if best_len >= max_k:
                return best_start, best_len

        return best_start, best_len

    # Strategy: Scan segments
    segment_size = 10**6
    
    regions = [
        10**12 - segment_size,
        5 * 10**11,
        10**11,
    ]
    
    random.seed(42)
    for _ in range(20):
        s = random.randint(10**9, 10**12 - segment_size)
        regions.append(s)
        
    global_best_k = 1
    global_best_s = 1
    
    for start in regions:
        if time.time() > deadline - 5:
            break
            
        if start < 1 or start + segment_size - 1 > 10**12:
            continue
            
        phi_list = compute_phi_range(start, segment_size)
        best_start_idx, best_len = find_max_run_in_phi_list(phi_list, 200)
        
        if best_len > global_best_k:
            global_best_k = best_len
            global_best_s = start + best_start_idx
            
        if global_best_k >= 200:
            return {'start': global_best_s, 'length': 200}
            
    if global_best_k < 2:
        return {'start': 2, 'length': 2}
        
    return {'start': global_best_s, 'length': global_best_k}
