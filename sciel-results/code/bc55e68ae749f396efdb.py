import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # 1. Generate primes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # 2. Segmented sieve for phi
    # We process segments of size SEG_SIZE
    SEG_SIZE = 2 * 10**6
    MAX_N = 10**12
    
    best_k = 1
    best_s = 1

    def find_longest_run_in_segment(phi_vals, offset):
        """
        Finds the longest run of distinct values in phi_vals.
        Returns (length, start_index_in_segment)
        """
        max_len = 0
        max_start = 0
        seen = set()
        current_start = 0
        
        for i, val in enumerate(phi_vals):
            if val in seen:
                # Collision found, reset the window
                seen.clear()
                current_start = i
            seen.add(val)
            current_len = i - current_start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
        
        return max_len, max_start

    current_start = 1
    while current_start <= MAX_N:
        if time_left() < 5.0: # Stop if we don't have enough time for a segment
            break
            
        current_end = min(current_start + SEG_SIZE, MAX_N + 1)
        segment_len = current_end - current_start
        
        # Initialize phi values for this segment
        # phi[i] initially holds the number itself
        phi_vals = [current_start + i for i in range(segment_len)]
        
        # Apply sieve steps for each prime
        for p in primes:
            if p * p > current_end:
                # We only need to divide by p if p is a factor.
                # Actually, the standard sieve for phi works by iterating primes up to sqrt(N) 
                # and dividing out p from multiples. 
                # But wait, phi(n) = n * product(1 - 1/p).
                # The segmented sieve approach:
                # For each prime p, find the first multiple in [start, end).
                # For each multiple m, phi[m] -= phi[m] // p.
                # This works for all primes p <= sqrt(end) ? 
                # No, it works for all primes p that divide any number in the segment.
                # If p > sqrt(end), it can divide at most one number in the segment (the number itself if it's prime, or a composite with a large factor).
                # However, the standard linear sieve or segmented sieve for phi usually requires primes up to N.
                # But we can't store primes up to 10^12.
                # The trick: 
                # phi(n) is computed correctly by dividing by all prime factors.
                # If we only iterate primes up to sqrt(N), we miss the large prime factors.
                # BUT, if we initialize phi[i] = i, and for each prime p <= sqrt(N), we do phi[m] -= phi[m]//p,
                # then for a number n, after processing all p <= sqrt(n), the remaining factor is either 1 or a prime > sqrt(n).
                # If it's a prime > sqrt(n), it appears with exponent 1.
                # So we must also handle the remaining factor.
                # In a segmented sieve, we can check if the remaining value > 1 and subtract it.
                
                # Let's refine the loop:
                # We iterate primes up to sqrt(MAX_N).
                # For each p, we update multiples.
                # After the loop, for each i in the segment, if phi_vals[i] > 1, 
                # it means there is a prime factor > sqrt(MAX_N) left.
                # Since p > sqrt(MAX_N), p can only divide n once (p^2 > MAX_N >= n).
                # So we do phi_vals[i] -= phi_vals[i] // phi_vals[i] ? No.
                # If remaining factor is R, then phi(n) = phi_temp * (R - 1) / R.
                # So phi_vals[i] -= phi_vals[i] // R.
                
                pass # This logic is handled after the prime loop

            # Find first multiple of p in [current_start, current_end)
            first_mult = ((current_start + p - 1) // p) * p
            if first_mult >= current_end:
                continue
                
            # Iterate over multiples
            for j in range(first_mult - current_start, segment_len, p):
                phi_vals[j] -= phi_vals[j] // p

        # Handle remaining prime factors > sqrt(MAX_N)
        # For each number, if phi_vals[i] > 1, it is the remaining prime factor.
        # Note: phi_vals[i] here is the value AFTER dividing by small primes.
        # Let n = i + current_start.
        # Let n = m * r, where m is product of small prime powers, r is 1 or prime > sqrt(N).
        # The sieve updates have effectively computed:
        # phi_temp = n * product_{p|m} (1 - 1/p)
        # We need phi(n) = phi_temp * (r - 1) / r if r > 1, else phi_temp.
        # So phi_vals[i] -= phi_vals[i] // phi_vals[i] is wrong.
        # It should be:
        # if phi_vals[i] > 1:
        #     phi_vals[i] -= phi_vals[i] // phi_vals[i]  <-- NO, this subtracts 1.
        #     Correct: phi(n) = phi_temp * (r-1)/r.
        #     So new_val = phi_temp - phi_temp // r.
        #     Here r = phi_vals[i] (the remaining factor).
        #     So phi_vals[i] -= phi_vals[i] // phi_vals[i] is phi_vals[i] - 1.
        #     Is r always the remaining factor?
        #     Yes, because all small prime factors have been divided out.
        #     The remaining part of the number is r.
        #     And phi(n) = phi(m*r) = phi(m)*phi(r) since gcd(m,r)=1.
        #     phi(m) is what we have in phi_vals[i] BEFORE this step?
        #     Wait. The sieve step `phi[j] -= phi[j] // p` updates the value in place.
        #     Initially phi[j] = n.
        #     After processing p, phi[j] = n * (1-1/p) ...
        #     So after all small primes, phi[j] = n * product_{p|n, p<=sqrt} (1-1/p).
        #     Let this be V.
        #     If n has a large prime factor r, then n = m * r.
        #     V = m * r * product_{p|m} (1-1/p) = phi(m) * r.
        #     We want phi(n) = phi(m) * phi(r) = phi(m) * (r-1).
        #     So we need to update V to V - V/r.
        #     Since V = phi(m)*r, V/r = phi(m).
        #     So V - V/r = phi(m)*r - phi(m) = phi(m)*(r-1). Correct.
        #     So the update is: phi_vals[i] -= phi_vals[i] // phi_vals[i] ?
        #     No, phi_vals[i] // phi_vals[i] is 1.
        #     We need to divide by the REMAINING FACTOR.
        #     But phi_vals[i] IS phi(m)*r.
        #     We don't know r directly from phi_vals[i] unless we stored it.
        #     However, we can just do:
        #     if phi_vals[i] > 1:
        #         # The remaining factor is not stored separately.
        #         # But wait, is phi_vals[i] equal to the remaining factor?
        #         # No.
        #         # Example: n=10. p=2,5.
        #         # Start 10.
        #         # p=2: 10 -> 10 - 5 = 5.
        #         # p=5: 5 -> 5 - 1 = 4.
        #         # Result 4. Correct.
        #         # Example: n=14. p=2,7.
        #         # Start 14.
        #         # p=2: 14 -> 7.
        #         # p=7: 7 -> 6.
        #         # Result 6. Correct.
        #         # Example: n=15. p=3,5.
        #         # Start 15.
        #         # p=3: 15 -> 10.
        #         # p=5: 10 -> 8.
        #         # Result 8. Correct.
        #         
        #         # What if n is prime? n=p > sqrt(N).
        #         # Start p.
        #         # No small primes divide it.
        #         # phi_vals[i] remains p.
        #         # We need phi(p) = p-1.
        #         # So if phi_vals[i] > 1, we must subtract 1?
        #         # No, if n=p, V=p. We want p-1.
        #         # If n=p*q (both large), impossible in segment if segment is small?
        #         # Actually, if n has two large prime factors, n > 10^12.
        #         # So n can have at most one prime factor > sqrt(N).
        #         # If it has one, say r, then n = m*r.
        #         # V = phi(m)*r.
        #         # We want phi(m)*(r-1) = V - phi(m).
        #         # But we don't have phi(m) directly.
        #         # However, V = phi(m)*r.
        #         # So phi(m) = V/r.
        #         # But we don't have r.
        #         # Wait, in the case n=p (prime), m=1, phi(m)=1, r=p. V=p.
        #         # V/r = 1. V - 1 = p-1. Correct.
        #         # In the case n=p*q? No, r is the remaining factor.
        #         # Is it true that V is divisible by the remaining factor?
        #         # Yes.
        #         # But how do we get the remaining factor?
        #         # We don't store it.
        #         # However, we can observe that for any n, after removing small primes,
        #         # the remaining factor r is either 1 or a prime.
        #         # If r=1, phi_vals[i] is already correct.
        #         # If r>1, phi_vals[i] = phi(m)*r.
        #         # We want to subtract phi(m).
        #         # phi(m) = phi_vals[i] / r.
        #         # We don't know r.
        #         
        #         # Let's look at the standard segmented phi algorithm.
        #         # Usually, one keeps an array `rem` or similar.
        #         # Or, simply:
        #         # The value `phi_vals[i]` after the sieve loop is NOT the final phi if there's a large factor.
        #         # Actually, the standard algorithm for segmented phi is:
        #         # 1. Init phi[i] = i.
        #         # 2. For each prime p, for multiples m, phi[m] -= phi[m]//p.
        #         # 3. This works for ALL primes p that divide m.
        #         # If we only iterate p up to sqrt(N), we miss large primes.
        #         # BUT, if a large prime p divides m, then m is either p (if m is prime) or m has another factor.
        #         # If m is prime, phi[m] remains m. We need m-1.
        #         # If m = p * k, and p > sqrt(N), then k < sqrt(N).
        #         # So k's prime factors are small.
        #         # The sieve will process k's factors.
        #         # Let's trace m = p * k.
        #         # phi[m] starts at p*k.
        #         # Factors of k are processed. phi[m] becomes p * phi(k).
        #         # Factor p is NOT processed (p > sqrt(N)).
        #         # So phi[m] ends up as p * phi(k).
        #         # We want phi(p) * phi(k) = (p-1) * phi(k) = p*phi(k) - phi(k).
        #         # So we need to subtract phi(k).
        #         # But phi[m] is p*phi(k).
        #         # We don't have phi(k) directly.
        #         
        #         # Correction: The standard segmented sieve for phi usually iterates primes up to sqrt(N) 
        #         # and then checks if the remaining value is > 1.
        #         # If remaining value > 1, it is a prime factor.
        #         # But the value stored in phi[i] is NOT the remaining factor.
        #         # The remaining factor is stored in a separate array or derived?
        #         
        #         # Actually, a simpler approach for large segments:
        #         # Just use trial division for the "remainder" part?
        #         # No, that's slow.
        #         
        #         # Let's use a different property.
        #         # If we iterate primes up to sqrt(N), we correctly compute phi(n) for all n
        #         # EXCEPT we fail to divide by large prime factors.
        #         # For a number n, let n = a * b where all prime factors of a are <= sqrt(N)
        #         # and b is either 1 or a prime > sqrt(N).
        #         # The sieve computes phi(a) * b.
        #         # We want phi(a) * (b-1).
        #         # Difference is phi(a).
        #         # So if b > 1, we need to subtract phi(a).
        #         # How to get phi(a)?
        #         # phi(a) = phi_current / b.
        #         # We don't know b.
        #         
        #         # However, we can store the "remaining part" of the number in a separate array `rem`.
        #         # Initialize rem[i] = i.
        #         # When we divide phi[i] by p, we also divide rem[i] by p as many times as possible?
        #         # No, just once for the first power?
        #         # Actually, for phi calculation, we just need to know IF p divides n.
        #         # So `rem[i]` can be the number with all small prime factors removed.
        #         # If rem[i] > 1, then rem[i] is the large prime factor b.
        #         # Then phi(a) = phi_current / rem[i].
        #         # And we subtract phi(a).
        #         
        #         # Let's implement this.
                pass

        # Re-implementing the segment loop with `rem` array
        # This part was inside the `for p in primes` loop, but needs to be separate.
        
        # Let's restart the segment computation logic properly.
        if time_left() < 5.0: break

        phi_vals = [current_start + i for i in range(segment_len)]
        rem_vals = [current_start + i for i in range(segment_len)]
        
        for p in primes:
            if p * p > current_end:
                break
                
            first_mult = ((current_start + p - 1) // p) * p
            if first_mult >= current_end:
                continue
                
            for j in range(first_mult - current_start, segment_len, p):
                phi_vals[j] -= phi_vals[j] // p
                while rem_vals[j] % p == 0:
                    rem_vals[j] //= p

        # Now handle the large prime factors
        for i in range(segment_len):
            if rem_vals[i] > 1:
                # rem_vals[i] is the large prime factor p_large
                # phi_vals[i] currently holds phi(a) * p_large
                # We want phi(a) * (p_large - 1)
                # phi(a) = phi_vals[i] // p_large
                phi_vals[i] -= phi_vals[i] // rem_vals[i]

        # Find longest run
        k, idx = find_longest_run_in_segment(phi_vals, current_start)
        if k > best_k:
            best_k = k
            best_s = current_start + idx
            # If we found a very long run, we might stop early?
            # Max k is 200. If we find 200, we can stop.
            if best_k >= 200:
                break

        current_start += SEG_SIZE

    return {'start': best_s, 'length': best_k}
