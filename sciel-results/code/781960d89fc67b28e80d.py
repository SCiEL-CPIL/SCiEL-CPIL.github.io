    # ... inside loop
        base = current_prod // old_p
        max_q = int(10**18 / base)
        
        # Filter candidates
        # We can optimize this by precomputing primes and using bisect
        # But 1000 primes is small.
        valid_candidates = [p for p in primes if p <= max_q and p != old_p]
        
        if not valid_candidates:
            continue
            
        q = random.choice(valid_candidates)
        new_prod = base * q
        # ...
