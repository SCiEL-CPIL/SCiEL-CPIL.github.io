        mask = rem % p == 0
        if np.any(mask):
            phi[mask] -= phi[mask] // p
            rem[mask] //= p
            # Check again?
            # For segmented sieve, usually we just divide once?
            # No, if n=4, p=2. rem=4. 4%2=0. phi-=phi//2. rem=2.
            # We need to remove the second 2?
            # The formula is phi(n) = n * prod (1-1/p).
            # We only apply (1-1/p) ONCE per distinct prime p.
            # So we don't need to remove all p's from rem for the phi update.
            # But we DO need to remove them from rem to correctly identify the "large prime" later?
            # If rem still has p, and we check for large prime, we might misidentify.
            # So yes, we must reduce rem to be coprime with all small primes.
        