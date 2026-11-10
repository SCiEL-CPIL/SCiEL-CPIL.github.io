import time
import sys

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX]."""
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction with look-ahead for large primes.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000  # Working range for greedy
    count = [0] * (L_MAX + 1)
    
    # Phase A: Assign small primes (p <= 100) to maximize coverage in [1, 2000]
    small_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
    
    for idx in small_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # Try all residues for small primes
        for r in range(p):
            start = p if r == 0 else r
            gain = 0
            # Maximize coverage of [1, 2000]
            for x in range(start, 2001, p):
                if count[x] == 1:
                    gain += 10 # Strong incentive to fix singles
                elif count[x] == 0:
                    gain += 1  # Weak incentive to start covering
                elif count[x] >= 2:
                    gain -= 0.5 # Penalty for over-covering
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Phase B: For larger primes, use look-ahead heuristic
    large_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] > 100]
    
    current_L = 0
    while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
        current_L += 1
    
    for idx in large_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        
        # Generate candidates: residues that cover gap and nearby
        candidates = set()
        for dx in range(-5, 6):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        
        # Also include current residue (0) and random-ish
        candidates.add(0)
        
        best_r = residues[idx]
        best_L = current_L
        
        # Evaluate candidates
        for r in candidates:
            # Simulate assignment
            start = p if r == 0 else r
            # We don't need to update full count array for every candidate if we are just estimating.
            # But for accuracy, let's do a quick check of the prefix extension.
            # Since we are adding one prime, we can just check if it helps the gap.
            # A more robust way: temporarily add, find L, remove.
            # This is O(L_MAX/p) per candidate. With ~10 candidates and ~150 primes, it's manageable.
            
            # Temporarily add
            for x in range(start, L_MAX + 1, p):
                count[x] += 1
            
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            # Remove
            for x in range(start, L_MAX + 1, p):
                count[x] -= 1
            
            if new_L > best_L:
                best_L = new_L
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        # Update current_L
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _local_search(residues, deadline, L_MAX=10000):
    """
    Targeted local search with single and double flips.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 20
    
    # Precompute positions for each prime for faster safety checks?
    # No, dynamic L makes it hard. We'll iterate on the fly.
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        best_move = None
        best_gain = 0
        
        # 1. Single Flip Search
        # Prioritize small primes as they have more flexibility
        for i in range(_NPRIMES):
            if time.time() > deadline:
                return residues, current_L
            
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Safety Check: Does removing current_r break any x in [1, current_L] with count[x] <= 2?
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate Flip
            new_start = p if desired_r == 0 else desired_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            # Undo
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        # 2. Double Flip Search (if no single flip found)
        if best_move is None:
            # Only consider small primes for double flips to keep it fast
            small_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 200]
            
            for i1 in range(len(small_indices)):
                if time.time() > deadline:
                    return residues, current_L
                i = small_indices[i1]
                p = _PRIMES[i]
                current_r1 = residues[i]
                desired_r1 = gap % p
                
                if current_r1 == desired_r1:
                    continue
                
                old_start1 = p if current_r1 == 0 else current_r1
                new_start1 = p if desired_r1 == 0 else desired_r1
                
                # Apply first flip
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] += 1
                
                for i2 in range(i1 + 1, len(small_indices)):
                    j = small_indices[i2]
                    q = _PRIMES[j]
                    current_r2 = residues[j]
                    desired_r2 = gap % q
                    
                    if current_r2 == desired_r2:
                        continue
                    
                    old_start2 = q if current_r2 == 0 else current_r2
                    new_start2 = q if desired_r2 == 0 else desired_r2
                    
                    # Check safety of second flip given first is applied
                    safe = True
                    for x in range(old_start2, current_L + 1, q):
                        if count[x] < 3:
                            safe = False
                            break
                    
                    if not safe:
                        continue
                    
                    # Apply second flip
                    for x in range(old_start2, L_MAX + 1, q):
                        count[x] -= 1
                    for x in range(new_start2, L_MAX + 1, q):
                        count[x] += 1
                    
                    new_L = current_L
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1
                    
                    # Undo second flip
                    for x in range(new_start2, L_MAX + 1, q):
                        count[x] -= 1
                    for x in range(old_start2, L_MAX + 1, q):
                        count[x] += 1
                    
                    gain = new_L - current_L
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, desired_r1, j, desired_r2, new_L)
                
                # Undo first flip
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] += 1
        
        if best_move is not None:
            if len(best_move) == 3:
                i, new_r, new_L = best_move
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                residues[i] = new_r
                current_L = new_L
            else:
                i, r1, j, r2, new_L = best_move
                p = _PRIMES[i]
                q = _PRIMES[j]
                old_r1 = residues[i]
                old_r2 = residues[j]
                old_start1 = p if old_r1 == 0 else old_r1
                new_start1 = p if r1 == 0 else r1
                old_start2 = q if old_r2 == 0 else old_r2
                new_start2 = q if r2 == 0 else r2
                
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] += 1
                for x in range(old_start2, L_MAX + 1, q):
                    count[x] -= 1
                for x in range(new_start2, L_MAX + 1, q):
                    count[x] += 1
                
                residues[i] = r1
                residues[j] = r2
                current_L = new_L
            
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 40)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Local search with moderate window
    if time.time() < deadline - 20:
        residues, L = _local_search(residues, deadline - 20, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Polish with larger window to extend further
    if time.time() < deadline - 5:
        residues, L = _local_search(best_residues, deadline - 5, L_MAX=20000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues
