"""Rewritten program: segmented-sieve based p-smooth run finder."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    # ---------- Sieve primes up to limit ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # ---------- Find p-smooth run length for a given p ----------
    def find_run_length(p, primes_all, prime_count, max_k=200):
        """
        primes_all: full list of primes (sorted)
        prime_count: number of primes to use (primes_all[0:prime_count] are <= p)
        """
        lo = p * p + 1
        width = max_k
        rem = [lo + i for i in range(width)]
        
        for idx in range(prime_count):
            q = primes_all[idx]
            # Find multiples of q in [lo, lo+width)
            # Number of multiples is ~ width/q
            # For q > width, at most 1 multiple
            first = ((lo + q - 1) // q) * q
            if first < lo:
                first += q
            if first >= lo + width:
                continue
            for j in range(first, lo + width, q):
                idx2 = j - lo
                while rem[idx2] % q == 0:
                    rem[idx2] //= q
        
        k = 0
        for i in range(width):
            if rem[i] == 1:
                k += 1
            else:
                break
        return k

    # ---------- Main search ----------
    best_k = 0
    best_p = 2

    SEARCH_LIMIT = 5000000
    primes_all = sieve_primes(SEARCH_LIMIT)
    
    n_primes = len(primes_all)
    
    # Process primes in chunks
    chunk_size = 500
    for start_idx in range(0, n_primes, chunk_size):
        if time.time() - start_time > DEADLINE - 5:
            break
        end_idx = min(start_idx + chunk_size, n_primes)
        for i in range(start_idx, end_idx):
            p = primes_all[i]
            k = find_run_length(p, primes_all, i + 1, max_k=200)
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
