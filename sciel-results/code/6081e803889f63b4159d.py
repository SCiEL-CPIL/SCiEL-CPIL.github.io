"""Doubly-covered prefix via simulated annealing with targeted gap repair."""
import time
import random
import math

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

def _greedy_construct(deadline):
    """
    Greedy construction: assign residues to maximize coverage density.
    """
    residues = [0] * _NPRIMES
    L_MAX = 2000
    count = [0] * (L_MAX + 1)
    
    # Process primes in increasing order
    for idx in range(_NPRIMES):
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1e9
        
        # Try a subset of residues for efficiency
        if p <= 50:
            candidate_rs = list(range(p))
        else:
            # Sample residues
            candidate_rs = [random.randint(0, p - 1) for _ in range(min(20, p))]
            candidate_rs.append(0)
            candidate_rs.append(1)
        
        for r in candidate_rs:
            start = p if r == 0 else r
            gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] == 0:
                    gain += 10
                elif count[x] == 1:
                    gain += 5
                elif count[x] == 2:
                    gain += 1
                else:
                    gain -= 1
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Compute current L
    current_L = 0
    while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
        current_L += 1
    
    return residues, current_L

def _simulated_annealing(residues, deadline, L_MAX=3000, T0=100.0, cooling=0.9995, max_steps=50000):
    """
    Simulated annealing to optimize the prefix length.
    Objective: maximize L (largest prefix with full double coverage).
    Moves: change one residue at a time.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    best_L = current_L
    best_residues = list(residues)
    
    T = T0
    step = 0
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Choose a prime to modify
        # Bias toward small primes (higher leverage)
        if random.random() < 0.7:
            idx = random.randint(0, min(30, _NPRIMES - 1))
        else:
            idx = random.randint(0, _NPRIMES - 1)
        
        p = _PRIMES[idx]
        old_r = residues[idx]
        
        # Choose new residue
        if random.random() < 0.5:
            # Targeted: try to cover the current gap
            gap = current_L + 1
            if gap <= L_MAX:
                new_r = gap % p
            else:
                new_r = random.randint(0, p - 1)
        else:
            # Random
            new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply the move
        old_start = p if old_r == 0 else old_r
        new_start = p if new_r == 0 else new_r
        
        for x in range(old_start, L_MAX + 1, p):
            count[x] -= 1
        for x in range(new_start, L_MAX + 1, p):
            count[x] += 1
        
        # Compute new L
        new_L = _find_prefix_L(count, L_MAX)
        
        # Accept or reject
        if new_L >= current_L:
            # Always accept improvements or equal
            residues[idx] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = list(residues)
        else:
            # Reject or accept with probability exp(-(delta)/T)
            delta = current_L - new_L
            if delta == 0:
                # Equal L: accept with moderate probability
                if random.random() < 0.3:
                    residues[idx] = new_r
                    current_L = new_L
                else:
                    # Undo
                    for x in range(new_start, L_MAX + 1, p):
                        count[x] -= 1
                    for x in range(old_start, L_MAX + 1, p):
                        count[x] += 1
            else:
                # Worse: accept with probability exp(-delta/T)
                prob = math.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    residues[idx] = new_r
                    current_L = new_L
                else:
                    # Undo
                    for x in range(new_start, L_MAX + 1, p):
                        count[x] -= 1
                    for x in range(old_start, L_MAX + 1, p):
                        count[x] += 1
        
        # Cool down
        T *= cooling
    
    return best_residues, best_L

def _targeted_gap_repair(residues, deadline, L_MAX=5000):
    """
    Targeted repair: for each gap position, try to find a prime whose residue
    can be changed to cover it without breaking existing coverage.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 20
    
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
        
        # Try to fix the gap by changing one prime's residue
        best_move = None
        best_new_L = current_L
        
        # Prioritize small primes
        for i in range(_NPRIMES):
            if time.time() > deadline:
                break
            
            p = _PRIMES[i]
            # Skip very large primes if time is tight
            if time.time() > deadline - 10 and p > 400:
                break
            
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Check if this change is safe
            old_start = p if current_r == 0 else current_r
            new_start = p if desired_r == 0 else desired_r
            
            # Check that no position in [1, current_L] drops below 2
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            new_L = _find_prefix_L(count, L_MAX)
            
            # Undo
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = (i, desired_r)
        
        if best_move is not None:
            i, new_r = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            residues[i] = new_r
            current_L = best_new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _multi_restart_sa(deadline, n_restarts=3):
    """
    Run simulated annealing from multiple different starting points.
    """
    best_overall = None
    best_L_overall = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a different starting point for each restart
        if restart == 0:
            # Use greedy construction
            residues, L = _greedy_construct(deadline - 60)
        else:
            # Use a more random start
            residues = [0] * _NPRIMES
            for i in range(_NPRIMES):
                p = _PRIMES[i]
                residues[i] = random.randint(0, p - 1)
            # Do a quick greedy pass to get a reasonable start
            L_MAX = 1000
            count = _build_count(residues, L_MAX)
            for idx in range(_NPRIMES):
                if time.time() > deadline - 60:
                    break
                p = _PRIMES[idx]
                best_r = residues[idx]
                best_gain = -1e9
                for r in range(min(10, p)):
                    start = p if r == 0 else r
                    gain = 0
                    for x in range(start, L_MAX + 1, p):
                        if count[x] <= 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                residues[idx] = best_r
                start = p if best_r == 0 else best_r
                for x in range(start, L_MAX + 1, p):
                    count[x] += 1
        
        # Run SA with remaining time
        time_per_restart = max(5, (deadline - time.time()) / max(1, n_restarts - restart))
        residues, L = _simulated_annealing(residues, time.time() + time_per_restart, L_MAX=3000)
        
        # Follow up with targeted repair
        if time.time() < deadline - 10:
            residues, L = _targeted_gap_repair(residues, deadline - 10, L_MAX=5000)
        
        if L > best_L_overall:
            best_L_overall = L
            best_overall = list(residues)
    
    return best_overall, best_L_overall

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Phase 1: Multi-restart SA
    best_residues, best_L = _multi_restart_sa(deadline - 20)
    
    # Phase 2: Final polish with larger range
    if time.time() < deadline - 5 and best_residues is not None:
        residues, L = _targeted_gap_repair(best_residues, deadline - 5, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification
    if best_residues is not None:
        final_L = _score(best_residues)
        if final_L > best_L:
            best_L = final_L
    
    if best_residues is None:
        # Fallback: all zeros
        best_residues = [0] * _NPRIMES
    
    return best_residues

# EVOLVE-BLOCK-END
