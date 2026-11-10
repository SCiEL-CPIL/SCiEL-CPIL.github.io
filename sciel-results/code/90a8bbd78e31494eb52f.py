"""Optimized doubly-covered prefix: NumPy-accelerated greedy + aggressive local search."""
import time
import numpy as np

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

def _build_count_np(residues, L_MAX):
    """Build count array using NumPy for speed."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        count[start-1::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # count is 0-indexed, so count[x-1] corresponds to x
    for x in range(1, L_MAX + 1):
        if count[x-1] < 2:
            return x - 1
    return L_MAX

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct_v2(deadline):
    """
    Improved greedy: process all primes, using coverage-deficit heuristic.
    Small primes get full search; larger primes use targeted candidates.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    # Process primes in order, but prioritize small primes first for dense coverage
    # Use a two-phase approach: first small primes (p <= 100), then larger
    small_indices = [i for i, p in enumerate(_PRIMES) if p <= 100]
    large_indices = [i for i, p in enumerate(_PRIMES) if p > 100]
    
    # Phase 1: Small primes - full search over all residues
    for i in small_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[i]
        best_r = 0
        best_gain = -1
        
        for r in range(p):
            start = p if r == 0 else r
            # Count how many positions with count==0 or count==1 we cover
            positions = range(start-1, L_MAX, p)
            gain = 0
            for x in positions:
                c = count[x]
                if c == 0:
                    gain += 3
                elif c == 1:
                    gain += 2
                elif c == 2:
                    gain += 0.5
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        count[start-1::p] += 1
    
    # Phase 2: Larger primes - targeted search around gaps
    for i in large_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[i]
        
        # Find first gap
        gap_idx = None
        for x in range(1, L_MAX + 1):
            if count[x-1] < 2:
                gap_idx = x
                break
        
        if gap_idx is None:
            residues[i] = 0
            continue
        
        # Candidate residues: focus on covering the gap and nearby under-covered positions
        candidates = set()
        candidates.add(gap_idx % p)
        
        # Add residues for positions near the gap
        for dx in range(-20, 21):
            x = gap_idx + dx
            if 1 <= x <= L_MAX:
                candidates.add(x % p)
        
        # Also try a few random candidates for diversity
        for _ in range(5):
            candidates.add(np.random.randint(p))
        
        best_r = gap_idx % p
        best_score = -1
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            
            # Score: weighted sum of coverage improvement
            positions = range(start-1, L_MAX, p)
            score = 0
            for x in positions:
                c = count[x]
                if c == 0:
                    score += 5
                elif c == 1:
                    score += 3
                elif c == 2:
                    score += 0.1
                else:
                    score -= 0.1  # Slight penalty for over-coverage
            
            # Strong bonus for covering the gap
            if (gap_idx - 1) % p == (start - 1) % p:
                score += 100
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        count[start-1::p] += 1
    
    current_L = _find_prefix_L_np(count, L_MAX)
    return residues, current_L

def _local_search_aggressive(residues, deadline, L_MAX=5000):
    """
    Aggressive local search: try all primes, not just bottleneck ones.
    Allow trades that temporarily reduce coverage if they enable longer prefix.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline:
        if stall_count > max_stall:
            break
        
        if current_L >= L_MAX:
            break
        
        gap = current_L + 1
        gap_idx = gap - 1  # 0-indexed
        
        # Check if gap is already covered
        if count[gap_idx] >= 2:
            # Extend prefix
            while gap <= L_MAX and count[gap-1] >= 2:
                current_L = gap
                gap += 1
            stall_count = 0
            continue
        
        # Try modifying each prime to see if we can extend the prefix
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            current_r = residues[i]
            
            # Try a set of candidate residues
            # 1. Residue that covers the gap
            # 2. Neighbors of current residue
            # 3. A few random candidates
            candidates = set()
            candidates.add(gap % p)
            candidates.add((current_r + 1) % p)
            candidates.add((current_r - 1) % p)
            if p > 10:
                candidates.add((current_r + p // 2) % p)
                candidates.add((current_r + p // 4) % p)
            
            for r in candidates:
                if r == current_r:
                    continue
                
                old_start = p if current_r == 0 else current_r
                new_start = p if r == 0 else r
                
                # Simulate change
                count[old_start-1::p] -= 1
                count[new_start-1::p] += 1
                
                # Check if prefix extends
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L] >= 2:
                    new_L += 1
                
                # Undo
                count[new_start-1::p] -= 1
                count[old_start-1::p] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, r, new_L)
        
        if best_move is None:
            stall_count += 1
            # Try random perturbations to escape local optima
            if stall_count % 10 == 0:
                for _ in range(3):
                    i = np.random.randint(_NPRIMES)
                    p = _PRIMES[i]
                    old_r = residues[i]
                    new_r = np.random.randint(p)
                    if new_r == old_r:
                        continue
                    
                    old_start = p if old_r == 0 else old_r
                    new_start = p if new_r == 0 else new_r
                    
                    count[old_start-1::p] -= 1
                    count[new_start-1::p] += 1
                    
                    new_L = _find_prefix_L_np(count, L_MAX)
                    if new_L > current_L:
                        residues[i] = new_r
                        current_L = new_L
                        stall_count = 0
                        break
                    else:
                        # Revert
                        count[new_start-1::p] -= 1
                        count[old_start-1::p] += 1
            continue
        
        i, new_r, new_L = best_move
        p = _PRIMES[i]
        old_r = residues[i]
        old_start = p if old_r == 0 else old_r
        new_start = p if new_r == 0 else new_r
        
        count[old_start-1::p] -= 1
        count[new_start-1::p] += 1
        
        residues[i] = new_r
        current_L = new_L
        stall_count = 0
    
    return residues, current_L

def _multi_start_search(residues, deadline, L_MAX=4000, n_starts=3):
    """
    Multi-start local search: perturb initial solution and run local search from each.
    """
    best_residues = list(residues)
    best_L = _find_prefix_L_np(_build_count_np(residues, L_MAX), L_MAX)
    
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        # Create perturbed version
        perturbed = list(residues)
        n_perturb = 5 + start * 3
        for _ in range(n_perturb):
            i = np.random.randint(_NPRIMES)
            p = _PRIMES[i]
            perturbed[i] = np.random.randint(p)
        
        # Run local search from perturbed state
        res, L = _local_search_aggressive(perturbed, deadline - (start * 10), L_MAX)
        
        if L > best_L:
            best_L = L
            best_residues = list(res)
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct_v2(deadline - 80)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Aggressive local search
    if time.time() < deadline - 40:
        residues, L = _local_search_aggressive(residues, deadline - 40, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Multi-start search
    if time.time() < deadline - 15:
        residues, L = _multi_start_search(best_residues, deadline - 15, L_MAX=4000, n_starts=3)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Final local search with larger range
    if time.time() < deadline - 10:
        residues, L = _local_search_aggressive(best_residues, deadline - 10, L_MAX=6000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification with full range
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_residues]

# EVOLVE-BLOCK-END
