N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Step 1: Greedy Sieve (Current Approach)
    S = list(range(N + 1))
    changed = True
    while changed:
        changed = False
        k = len(S)
        primes_to_check = _primes(k)
        for p in primes_to_check:
            classes = [0] * p
            for v in S:
                classes[v % p] += 1
            if min(classes) > 0:
                min_count = min(classes)
                target_residue = classes.index(min_count)
                S = [v for v in S if v % p != target_residue]
                changed = True
                break
    
    # Step 2: Add-back Heuristic
    # Maintain state for admissibility check
    # We need to know for each prime p <= len(S), which residues are present.
    # To allow adding back, we need to track counts.
    
    # Re-initialize state for the current S
    current_size = len(S)
    # We only care about primes up to current_size + some margin for additions
    # Max size is N+1 = 5001.
    all_primes = _primes(N + 1)
    
    # residue_counts[p] = list of counts for residues 0..p-1
    # We only track primes p <= current_size initially, but as size grows, we need more.
    # For efficiency, we can just check admissibility from scratch if N is small enough?
    # N=5000. Checking all primes <= 5000 for each addition is too slow.
    
    # Optimized Check:
    # Maintain a dictionary or list of counts for relevant primes.
    # When adding x, for each prime p <= new_size:
    #   r = x % p
    #   if counts[p][r] == 0:
    #       counts[p][r] = 1
    #       if all(c > 0 for c in counts[p]):
    #           # Conflict!
    #           return False
    #   else:
    #       counts[p][r] += 1
    #   # If counts[p][r] was 0 and now 1, and this makes all >0, it's bad.
    #   # Actually, if it was 0, and now 1, we must check if OTHERS are >0.
    
    # Let's store counts for all primes <= N+1.
    # Memory: Sum of p for p <= 5000 is approx 5000^2 / 2 / ln(5000) ~ 10M integers. OK.
    
    counts = {}
    for p in all_primes:
        counts[p] = [0] * p
    
    # Populate counts with current S
    for v in S:
        for p in all_primes:
            if p > len(S): break # Only need primes <= len(S) for validity, but we might add more.
            # Actually, if p > len(S), it's impossible to cover all residues.
            # So we only need to track p <= len(S).
            # But as we add, len(S) increases.
            # So we should track all p <= N+1 to be safe, or update dynamically.
            # Let's just track all p <= N+1.
            counts[p][v % p] += 1
            
    # Helper to check if adding x is valid
    def can_add(x):
        new_size = len(S) + 1
        # Check primes p <= new_size
        # We need to ensure that for NO prime p <= new_size do we cover all residues.
        # Adding x only increases counts.
        # So we only need to check if x completes a set for any p.
        for p in all_primes:
            if p > new_size:
                break
            r = x % p
            if counts[p][r] == 0:
                # This residue was missing. Now it's present.
                # Check if all other residues are present.
                # If yes, adding x makes it invalid.
                # If no, it's still valid (some other residue is still missing).
                is_complete = True
                for i in range(p):
                    if i != r and counts[p][i] == 0:
                        is_complete = False
                        break
                if is_complete:
                    return False
            # If counts[p][r] > 0, adding x doesn't change the "completeness" status 
            # (it was already complete or incomplete, and adding to a non-zero bin 
            # doesn't make a zero bin non-zero).
            # Wait, if it was already complete, it's already invalid?
            # No, S is currently valid. So no p <= len(S) is complete.
            # So if counts[p][r] > 0, it means that residue was already present.
            # The set was NOT complete before.
            # Adding x doesn't fill any new holes.
            # So it remains not complete.
            # So only the case where counts[p][r] == 0 is dangerous.
            
        return True

    # Try to add back removed numbers
    removed = [v for v in range(N + 1) if v not in S]
    # Sort removed? Maybe smallest first?
    # Or random?
    # Let's try smallest first.
    for x in removed:
        if can_add(x):
            S.append(x)
            # Update counts
            for p in all_primes:
                if p > len(S): break
                counts[p][x % p] += 1

    return S
