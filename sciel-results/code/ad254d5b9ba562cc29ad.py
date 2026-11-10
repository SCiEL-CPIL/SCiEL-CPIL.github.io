"""Optimized doubly-covered prefix: constraint-satisfaction approach with targeted local search."""
import time

# EVOLVE-BLOCK-START

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

def _coverage_deficit(count, L_MAX, x):
    """Return how many additional covers x needs (0 if already >= 2)."""
    return max(0, 2 - count[x])

def _greedy_csp_construct(deadline):
    """
    Constraint-satisfaction-style greedy:
    1. Fix small primes (p <= 50) to maximize coverage density in [1, 2000].
    2. For larger primes, use coverage-deficit heuristic to repair first uncovered.
    """
    residues = [0] * _NPRIMES
    L_MAX = 3000
    count = [0] * (L_MAX + 1)
    
    # Phase 1: Fix small primes for dense coverage
    # Small primes cover many positions; choose residues to maximize overlap in [1, 2000]
    small_primes = [(i, p) for i, p in enumerate(_PRIMES) if p <= 50]
    large_primes = [(i, p) for i, p in enumerate(_PRIMES) if p > 50]
    
    # For small primes, try to cover positions 1..2000 densely
    # Strategy: for each small prime, choose residue that covers the most under-covered positions
    for i, p in small_primes:
        if time.time() > deadline:
            break
        best_r = 0
        best_gain = -1
        
        for r in range(p):
            start = p if r == 0 else r
            gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] == 0:
                    gain += 2  # High value for covering uncovered
                elif count[x] == 1:
                    gain += 1  # Medium value for reaching double coverage
                # No penalty for over-covering in this phase
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Phase 2: For larger primes, use coverage-deficit heuristic
    # Process in order of increasing prime size
    for i, p in large_primes:
        if time.time() > deadline:
            break
        
        # Find the first position in [1, L_MAX] that needs more coverage
        gap = None
        for x in range(1, L_MAX + 1):
            if count[x] < 2:
                gap = x
                break
        
        if gap is None:
            # All positions covered; just pick a residue that doesn't hurt
            residues[i] = 0
            continue
        
        # Try residues that cover the gap, preferring those that also help nearby under-covered positions
        best_r = gap % p
        best_score = -1
        
        # Candidate residues: those that cover gap, plus some alternatives
        candidates = set()
        candidates.add(gap % p)
        # Also try residues that cover positions near the gap
        for dx in range(-10, 11):
            x = gap + dx
            if 1 <= x <= L_MAX:
                candidates.add(x % p)
        
        for r in candidates:
            if r >= p:
                continue
            
            # Score: how many under-covered positions does this residue cover?
            start = p if r == 0 else r
            score = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] == 0:
                    score += 3
                elif count[x] == 1:
                    score += 2
                # Slight penalty for positions far from gap
                dist = abs(x - gap)
                if dist > 100:
                    score -= 0.1
            
            # Bonus for covering the gap itself
            if gap % p == r:
                score += 50
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Find current prefix L
    current_L = _find_prefix_L(count, L_MAX)
    return residues, current_L

def _identify_bottleneck_primes(residues, count, L, L_MAX):
    """
    Identify primes that cover the bottleneck position L+1.
    These are the primes whose residue change could extend the prefix.
    """
    gap = L + 1
    if gap > L_MAX:
        return []
    
    bottleneck = []
    for i, p in enumerate(_PRIMES):
        # Does prime i cover the gap?
        if gap % p == residues[i]:
            bottleneck.append(i)
    
    # Also include primes that are close to covering the gap
    # (within 2 positions) for potential future coverage
    for i, p in enumerate(_PRIMES):
        if i in bottleneck:
            continue
        # Check if changing this prime's residue would cover the gap
        # and if it's a small prime (more frequent)
        if p <= 100:
            bottleneck.append(i)
    
    return bottleneck

def _local_search_targeted(residues, deadline, L_MAX=3000):
    """
    Targeted local search: only modify primes that cover the bottleneck.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline:
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            # Extend prefix
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            stall_count = 0
            continue
        
        # Identify bottleneck primes
        bottleneck = _identify_bottleneck_primes(residues, count, current_L, L_MAX)
        
        if not bottleneck:
            break
        
        best_move = None
        best_gain = 0
        
        for i in bottleneck:
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Check safety: all positions in [1, current_L] that lose coverage must have count >= 3
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate the flip
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
        
        if best_move is None:
            stall_count += 1
            continue
        
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
        stall_count = 0
    
    return residues, current_L

def _polish_pass(residues, deadline, L_MAX=5000):
    """
    Final polish: try small adjustments to extend coverage further.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    # Try each prime once with a few alternative residues
    for i in range(_NPRIMES):
        if time.time() > deadline:
            break
        
        p = _PRIMES[i]
        current_r = residues[i]
        
        # Try a few alternative residues
        alternatives = set()
        alternatives.add((current_r + 1) % p)
        alternatives.add((current_r - 1) % p)
        alternatives.add((current_r + p // 2) % p)
        
        best_r = current_r
        best_L = current_L
        
        for r in alternatives:
            if r == current_r:
                continue
            
            # Check safety
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate
            new_start = p if r == 0 else r
            
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
            
            if new_L > best_L:
                best_L = new_L
                best_r = r
        
        if best_r != current_r:
            old_start = p if current_r == 0 else current_r
            new_start = p if best_r == 0 else best_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            residues[i] = best_r
            current_L = best_L
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: CSP-style greedy construction
    residues, L = _greedy_csp_construct(deadline - 60)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Targeted local search on bottleneck primes
    if time.time() < deadline - 30:
        residues, L = _local_search_targeted(residues, deadline - 30, L_MAX=3000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Polish pass with larger range
    if time.time() < deadline - 15:
        residues, L = _polish_pass(best_residues, deadline - 15, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Another targeted local search if time permits
    if time.time() < deadline - 10:
        residues, L = _local_search_targeted(best_residues, deadline - 10, L_MAX=4000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification with full range
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
