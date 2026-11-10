import time
import random

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
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _extend_L(count, current_L, L_MAX):
    """Extend L forward as far as possible."""
    L = current_L
    while L < L_MAX and count[L + 1] >= 2:
        L += 1
    return L

def _shrink_L(count, current_L, min_check=1):
    """Find new L by scanning backward from current_L-1 until we find a gap."""
    L = current_L
    while L >= min_check and count[L] >= 2:
        L -= 1
    return L

def _greedy_repair(residues, count, current_L, L_MAX, deadline):
    """Try to fix the first gap by moving one prime's residue."""
    if current_L >= L_MAX:
        return residues, current_L
    
    gap = current_L + 1
    if count[gap] >= 2:
        return _extend_L(count, current_L, L_MAX) and (residues, _extend_L(count, current_L, L_MAX))
    
    # Try all primes, prefer small ones
    best_idx = -1
    best_loss = float('inf')
    
    for i in range(_NPRIMES):
        if time.time() > deadline - 5:
            break
        p = _PRIMES[i]
        desired_r = gap % p
        if residues[i] == desired_r:
            continue
        
        old_r = residues[i]
        # Estimate loss: how many positions in [1, current_L] have count exactly 2
        # and would lose coverage if we change this prime
        loss = 0
        start_old = p if old_r == 0 else old_r
        for x in range(start_old, current_L + 1, p):
            if count[x] == 2:
                loss += 1
        
        # Prefer moves with minimal loss, then smaller primes
        if loss < best_loss or (loss == best_loss and best_idx != -1 and p < _PRIMES[best_idx]):
            best_loss = loss
            best_idx = i
    
    if best_idx != -1:
        i = best_idx
        p = _PRIMES[i]
        old_r = residues[i]
        new_r = gap % p
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        for x in range(start_old, L_MAX + 1, p):
            count[x] -= 1
        for x in range(start_new, L_MAX + 1, p):
            count[x] += 1
        
        residues[i] = new_r
        current_L = _extend_L(count, current_L, L_MAX)
    
    return residues, current_L

def _sa_optimize(residues, count, current_L, L_MAX, deadline, seed=None):
    """Simulated annealing with fast L updates."""
    if seed is not None:
        random.seed(seed)
    
    best_residues = list(residues)
    best_L = current_L
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.001
    stall_counter = 0
    max_stall = 500
    
    while time.time() < deadline - 10 and T > min_T:
        # Pick a prime to modify
        # Bias toward small primes (more frequent) and primes that cover the gap
        i = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[i]
        old_r = residues[i]
        
        # Choose new residue: 50% random, 50% targeted at gap
        if random.random() < 0.5 and current_L < L_MAX:
            new_r = (current_L + 1) % p
        else:
            new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        # Update counts
        for x in range(start_old, L_MAX + 1, p):
            count[x] -= 1
        for x in range(start_new, L_MAX + 1, p):
            count[x] += 1
        
        residues[i] = new_r
        
        # Fast L update
        if count[current_L] >= 2:
            # Try to extend
            new_L = _extend_L(count, current_L, L_MAX)
        else:
            # Need to shrink
            new_L = _shrink_L(count, current_L)
        
        delta = new_L - current_L
        
        if new_L > best_L:
            best_L = new_L
            best_residues = list(residues)
            stall_counter = 0
        elif delta > 0:
            stall_counter = 0
        else:
            stall_counter += 1
        
        # Acceptance
        if delta >= 0:
            current_L = new_L
        else:
            prob = 2.718281828 ** (delta / T)
            if random.random() > prob:
                # Revert
                for x in range(start_new, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(start_old, L_MAX + 1, p):
                    count[x] += 1
                residues[i] = old_r
                # current_L unchanged
            else:
                current_L = new_L
        
        T *= cooling
        
        # If stalled for too long, try a greedy repair
        if stall_counter >= max_stall:
            stall_counter = 0
            residues, current_L = _greedy_repair(residues, count, current_L, L_MAX, deadline)
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # Multi-start approach
    best_residues = None
    best_L = 0
    L_MAX = 15000  # Working window for SA
    
    # Initial greedy construction: start with all 1s, then repair
    base_residues = [1 % p for p in _PRIMES]
    count = _build_count(base_residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    # Greedy repair phase
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            current_L = _extend_L(count, current_L, L_MAX)
            improved = True
            continue
        
        # Try to fix gap
        best_idx = -1
        best_loss = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            desired_r = gap % p
            if base_residues[i] == desired_r:
                continue
            
            old_r = base_residues[i]
            loss = 0
            start_old = p if old_r == 0 else old_r
            for x in range(start_old, current_L + 1, p):
                if count[x] == 2:
                    loss += 1
            
            if loss < best_loss:
                best_loss = loss
                best_idx = i
        
        if best_idx != -1 and best_loss < 5:  # Only accept if loss is small
            i = best_idx
            p = _PRIMES[i]
            old_r = base_residues[i]
            new_r = gap % p
            
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            for x in range(start_old, L_MAX + 1, p):
                count[x] -= 1
            for x in range(start_new, L_MAX + 1, p):
                count[x] += 1
            
            base_residues[i] = new_r
            current_L = _extend_L(count, current_L, L_MAX)
            improved = True
    
    if current_L > best_L:
        best_L = current_L
        best_residues = list(base_residues)
    
    # Now do SA with multiple restarts
    remaining_time = deadline - 30 - time.time()
    num_restarts = max(1, int(remaining_time / 25))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # For restart > 0, perturb the best solution
        if restart == 0:
            start_residues = list(best_residues)
        else:
            start_residues = list(best_residues)
            # Perturb 10-20 random residues
            num_perturb = random.randint(10, 20)
            for _ in range(num_perturb):
                idx = random.randint(0, _NPRIMES - 1)
                start_residues[idx] = random.randint(0, _PRIMES[idx] - 1)
        
        count_r = _build_count(start_residues, L_MAX)
        L_r = _find_prefix_L(count_r, L_MAX)
        
        # SA on this starting point
        sub_deadline = min(deadline - 15, time.time() + 20)
        res_residues, res_L = _sa_optimize(start_residues, count_r, L_r, L_MAX, sub_deadline, seed=restart)
        
        if res_L > best_L:
            best_L = res_L
            best_residues = list(res_residues)
    
    # Final verification with larger window
    if best_residues is None:
        best_residues = [1 % p for p in _PRIMES]
    
    L_MAX_FINAL = 50000
    count_final = _build_count(best_residues, L_MAX_FINAL)
    final_L = _find_prefix_L(count_final, L_MAX_FINAL)
    
    return best_residues
