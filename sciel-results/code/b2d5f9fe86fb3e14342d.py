import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to a reasonable limit for fast factorization
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For Miller-Rabin, we need small primes for trial division first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime_mr(n):
        """Miller-Rabin primality test for 64-bit integers."""
        if n < 2:
            return False
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Witnesses sufficient for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1].
        Uses segmented sieve logic.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi values to the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # We also need to track the remaining part of the original number to detect large prime factors
        # rem[i] will hold the part of (start+i) not yet divided out by small primes
        rem = np.arange(start, start + length, dtype=np.int64)
        
        for p in primes:
            if p * p > start + length - 1:
                # Optimization: if p*p > max_val, we could stop? 
                # No, because a number like 2 * (large prime) needs the large prime handled.
                # But we only have primes up to 10^6.
                # If p > sqrt(max_val), then a number can have at most one such factor.
                # However, our primes list goes up to 10^6.
                # If start is around 10^12, sqrt is 10^6.
                # So we should process all primes in the list.
                pass
            
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
                
            # Vectorized update for phi
            # indices where the number is divisible by p
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Apply factor (1 - 1/p) = (p-1)/p
            # phi = phi // p * (p-1)
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Update rem: divide by p as many times as possible
            # This is the tricky part to vectorize efficiently without loops for each index.
            # Since length is small (e.g., 200-500), and p is prime,
            # the number of times p divides any specific number is small.
            # However, doing a python loop over indices for every prime might be slow.
            # Let's try a vectorized approach for the first division, then check for second.
            
            # First division
            rem[indices] //= p
            
            # Check for second division
            # rem[indices] % p == 0
            mask2 = rem[indices] % p == 0
            if np.any(mask2):
                idxs2 = indices[mask2]
                rem[idxs2] //= p
                
                # Check for third division (very rare for p > 2)
                mask3 = rem[idxs2] % p == 0
                if np.any(mask3):
                    idxs3 = idxs2[mask3]
                    rem[idxs3] //= p
                    # For p=2, could be higher, but unlikely for large numbers in a short range
                    # To be safe, we can do a small while loop for the few cases where rem is still divisible
                    # Or just loop for the specific indices if the count is very small
                    if p == 2:
                        for idx in idxs3:
                            while rem[idx] % 2 == 0:
                                rem[idx] //= 2
                    else:
                        # For odd primes, p^3 > 10^9, so p^4 > 10^12 is rare but possible for small p
                        # Actually 3^25 > 10^12. So for p=3, we might need more divisions.
                        # Let's just do a while loop for the remaining indices if p is small.
                        if p < 100:
                            for idx in idxs3:
                                while rem[idx] % p == 0:
                                    rem[idx] //= p
                        else:
                            # For large p, p^3 > 10^9, so at most 3 or 4 times.
                            # We already did 3.
                            mask4 = rem[idxs3] % p == 0
                            if np.any(mask4):
                                idxs4 = idxs3[mask4]
                                rem[idxs4] //= p

        # After processing all small primes, any remaining factor in rem[i] > 1 is a large prime
        # (since if it were composite, it would have a factor <= sqrt(rem[i]) <= sqrt(10^12) = 10^6, 
        # which we would have found and divided out).
        # Wait, is this true?
        # If rem[i] is composite, say a*b, with a<=b. Then a <= sqrt(rem[i]).
        # If a <= 10^6, we would have processed a.
        # So if a > 10^6, then a*b > 10^12, which is impossible since rem[i] <= 10^12.
        # So yes, any remaining factor > 1 is prime.
        
        mask = rem > 1
        if np.any(mask):
            # Apply factor (1 - 1/q) where q = rem[i]
            # phi = phi // q * (q-1)
            q_vals = rem[mask]
            phi_vals[mask] = (phi_vals[mask] // q_vals) * (q_vals - 1)
            
        return phi_vals

    def find_best_run(phis, start):
        """
        Find the longest contiguous subsequence of phis with distinct values.
        Returns (length, start_index_in_original_numbering)
        """
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(len(phis)):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Strategy:
    # 1. Check small numbers (1 to 1000) quickly.
    # 2. Use random sampling across the range [1, 10^12].
    # 3. Also try specific "structured" starts if time permits?
    #    The prompt suggests targeting numbers with unique prime factorization structures.
    #    Random sampling is effectively exploring this space.
    #    We can also try starts near primes?
    #    Let's stick to random sampling but ensure we cover a wide range.
    
    if time_left() > 5:
        phis = compute_phi_segment(1, 1000)
        k, s = find_best_run(phis, 1)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    seg_len = 200
    
    # We can increase seg_len if compute_phi_segment is fast enough.
    # Let's benchmark one segment first.
    test_start = 10**10
    t0 = time.time()
    _ = compute_phi_segment(test_start, 200)
    t1 = time.time()
    cost_per_seg = t1 - t0
    
    # Estimate how many segments we can do
    # We want to spend about 80% of time on search
    search_time = time_left() - 2
    max_segments = int(search_time / cost_per_seg) if cost_per_seg > 0 else 1000
    
    # Cap segments to avoid running out of time on overhead
    # Also, we can try larger segments if cost is low.
    # If cost is very low, we can increase seg_len to find longer runs.
    # But longer segments are more likely to contain duplicates?
    # Actually, longer segments give more chances to find a long run.
    # Let's try seg_len = 500 if cost allows.
    
    # Adjust seg_len based on cost
    if cost_per_seg < 0.01:
        seg_len = 500
    elif cost_per_seg < 0.05:
        seg_len = 300
    else:
        seg_len = 200

    # Re-estimate cost with new seg_len
    t0 = time.time()
    _ = compute_phi_segment(test_start, seg_len)
    t1 = time.time()
    cost_per_seg = t1 - t0
    
    max_segments = int(search_time / cost_per_seg) if cost_per_seg > 0 else 100
    max_segments = min(max_segments, 5000) # Cap to be safe

    for _ in range(max_segments):
        if time_left() < 2:
            break
            
        # Random start
        # Ensure s + seg_len - 1 <= 10^12
        max_start = 10**12 - seg_len
        if max_start <= 1:
            start = 1
        else:
            start = random.randint(1, max_start)
            
        phis = compute_phi_segment(start, seg_len)
        k, s = find_best_run(phis, start)
        if k > best_k:
            best_k, best_s = k, s
            
        if best_k >= 200:
            break

    return {'start': best_s, 'length': best_k}
