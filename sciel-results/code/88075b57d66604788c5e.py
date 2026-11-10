"""Optimized doubly-covered prefix: numpy-accelerated greedy + aggressive local search."""
import time
import numpy as np
import random

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
_PRIMES_ARR = np.array(_PRIMES, dtype=np.int64)

def _build_count_np(residues, L_MAX):
    """Build count array using numpy for speed."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # Find first position where count < 2
    below = np.where(count[1:] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return below[0]  # 0-indexed in count[1:], so this is the x value

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return int(_find_prefix_L_np(count, L_MAX))

def _greedy_construct(deadline):
    """
    Greedy construction: process primes in order, choosing residue to maximize
    coverage of under-covered positions, with emphasis on the current gap.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    prime_order = list(range(_NPRIMES))
    current_L = 0
    
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        
        # Generate diverse candidate residues
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        candidates.add(1)
        # Positions near the gap
        for dx in range(-20, 21):
            x = gap + dx
            if 1 <= x <= L_MAX:
                candidates.add(x % p)
        # Random candidates for diversity
        for _ in range(5):
            candidates.add(random.randint(0, p - 1))
        
        best_r = 0
        best_score = -1e18
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            if start > L_MAX:
                continue
            
            # Score: weighted sum of coverage improvements
            positions = np.arange(start, L_MAX + 1, p)
            c_vals = count[positions]
            
            # Gain from covering under-covered positions
            gain = np.sum(np.where(c_vals <= 1, 1, 0))
            # Penalty for over-covering
            penalty = np.sum(np.where(c_vals >= 3, 0.5, 0))
            # Bonus for covering the gap
            if r == gap % p:
                gain += 50
            # Bonus for covering positions just after the gap
            for dx in range(1, 6):
                if (gap + dx) % p == r:
                    gain += 10
            
            score = gain - penalty
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = int(best_r)
        
        # Update count
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            count[start::p] += 1
        
        # Update current_L
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _local_search(residues, deadline, L_MAX=5000):
    """
    Aggressive local search: try multiple candidate residues per prime,
    with relaxed safety checks and bottleneck-driven candidate generation.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = int(_find_prefix_L_np(count, L_MAX))
    
    improved = True
    stall_count = 0
    max_stall = 30
    
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
        
        # Need to increase count[gap]. Try flipping residues.
        best_move = None
        best_gain = 0
        
        # Generate candidates: primes that can cover the gap, plus others
        for i in range(_NPRIMES):
            if time.time() > deadline:
                break
            p = _PRIMES[i]
            current_r = residues[i]
            
            # Candidate residues: gap % p and nearby
            candidate_rs = set()
            candidate_rs.add(gap % p)
            for dx in range(-5, 6):
                x = gap + dx
                if 1 <= x <= L_MAX:
                    candidate_rs.add(x % p)
            # Also try a few random ones
            for _ in range(2):
                candidate_rs.add(random.randint(0, p - 1))
            
            for new_r in candidate_rs:
                if new_r == current_r or new_r >= p:
                    continue
                
                old_start = p if current_r == 0 else current_r
                new_start = p if new_r == 0 else new_r
                
                # Check safety: positions that lose coverage must have count >= 3
                if old_start <= current_L:
                    old_positions = np.arange(old_start, current_L + 1, p)
                    if np.any(count[old_positions] < 3):
                        continue
                
                # Simulate
                if old_start <= L_MAX:
                    count[old_start::p] -= 1
                if new_start <= L_MAX:
                    count[new_start::p] += 1
                
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                    new_L += 1
                
                # Undo
                if new_start <= L_MAX:
                    count[new_start::p] -= 1
                if old_start <= L_MAX:
                    count[old_start::p] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, int(new_r), new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            if old_start <= L_MAX:
                count[old_start::p] -= 1
            if new_start <= L_MAX:
                count[new_start::p] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _local_search_2flip(residues, deadline, L_MAX=5000):
    """
    2-flip local search: try pairs of residue flips to extend the doubly-covered prefix.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = int(_find_prefix_L_np(count, L_MAX))
    
    improved = True
    stall_count = 0
    max_stall = 15
    
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
        
        # Try pairs among small primes
        small_primes = [i for i in range(_NPRIMES) if _PRIMES[i] <= 150]
        
        for i1 in range(len(small_primes)):
            if time.time() > deadline:
                break
            i = small_primes[i1]
            p = _PRIMES[i]
            current_r1 = residues[i]
            desired_r1 = gap % p
            
            if current_r1 == desired_r1:
                continue
            
            old_start1 = p if current_r1 == 0 else current_r1
            new_start1 = p if desired_r1 == 0 else desired_r1
            
            # Check safety for first flip
            if old_start1 <= current_L:
                old_pos1 = np.arange(old_start1, current_L + 1, p)
                if np.any(count[old_pos1] < 3):
                    continue
            
            # Apply first flip
            if old_start1 <= L_MAX:
                count[old_start1::p] -= 1
            if new_start1 <= L_MAX:
                count[new_start1::p] += 1
            
            for i2 in range(i1 + 1, len(small_primes)):
                j = small_primes[i2]
                q = _PRIMES[j]
                current_r2 = residues[j]
                desired_r2 = gap % q
                
                if current_r2 == desired_r2:
                    continue
                
                old_start2 = q if current_r2 == 0 else current_r2
                new_start2 = q if desired_r2 == 0 else desired_r2
                
                # Check safety for second flip (with first already applied)
                if old_start2 <= current_L:
                    old_pos2 = np.arange(old_start2, current_L + 1, q)
                    if np.any(count[old_pos2] < 3):
                        continue
                
                # Apply second flip
                if old_start2 <= L_MAX:
                    count[old_start2::q] -= 1
                if new_start2 <= L_MAX:
                    count[new_start2::q] += 1
                
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                    new_L += 1
                
                # Undo second flip
                if new_start2 <= L_MAX:
                    count[new_start2::q] -= 1
                if old_start2 <= L_MAX:
                    count[old_start2::q] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, desired_r1, j, desired_r2, new_L)
            
            # Undo first flip
            if new_start1 <= L_MAX:
                count[new_start1::p] -= 1
            if old_start1 <= L_MAX:
                count[old_start1::p] += 1
        
        if best_move is not None:
            i, r1, j, r2, new_L = best_move
            p = _PRIMES[i]
            q = _PRIMES[j]
            old_r1 = residues[i]
            old_r2 = residues[j]
            old_start1 = p if old_r1 == 0 else old_r1
            new_start1 = p if r1 == 0 else r1
            old_start2 = q if old_r2 == 0 else old_r2
            new_start2 = q if r2 == 0 else r2
            
            if old_start1 <= L_MAX:
                count[old_start1::p] -= 1
            if new_start1 <= L_MAX:
                count[new_start1::p] += 1
            if old_start2 <= L_MAX:
                count[old_start2::q] -= 1
            if new_start2 <= L_MAX:
                count[new_start2::q] += 1
            
            residues[i] = int(r1)
            residues[j] = int(r2)
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _multi_start_local_search(residues, deadline, L_MAX=5000, n_starts=5):
    """
    Multi-start local search: run local search from multiple perturbed starting points.
    """
    best_residues = list(residues)
    best_L = int(_find_prefix_L_np(_build_count_np(residues, L_MAX), L_MAX))
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Perturb: randomly change a few residues
        perturbed = list(residues)
        n_changes = random.randint(2, 5)
        for _ in range(n_changes):
            idx = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[idx]
            perturbed[idx] = random.randint(0, p - 1)
        
        r, L = _local_search(perturbed, deadline - (n_starts - s) * 0.5, L_MAX)
        if L > best_L:
            best_L = L
            best_residues = list(r)
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit)
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 60)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Local search
    if time.time() < deadline - 30:
        residues, L = _local_search(residues, deadline - 30, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: 2-flip local search
    if time.time() < deadline - 15:
        residues, L = _local_search_2flip(best_residues, deadline - 15, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Multi-start local search
    if time.time() < deadline - 10:
        residues, L = _multi_start_local_search(best_residues, deadline - 10, L_MAX=5000, n_starts=3)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 3:
        residues, L = _local_search(best_residues, deadline - 3, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification with full range
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    # Ensure all residues are Python integers
    return [int(r) for r in best_residues]

# EVOLVE-BLOCK-END
