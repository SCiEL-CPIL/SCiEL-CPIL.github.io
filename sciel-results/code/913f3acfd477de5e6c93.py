"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        counts[start::p] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find prefix length from counts array."""
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Working range for optimization - extend to 5000 for better search
    WORK_L = 5000
    
    # Phase 1: Smart initialization
    residues = [0] * n_primes
    counts = np.zeros(WORK_L + 1, dtype=np.int32)
    
    # Greedy initialization: for each prime, choose residue that maximizes
    # coverage of under-covered positions
    order = list(range(n_primes))
    order.sort(key=lambda i: primes[i])
    
    for idx in order:
        p = primes[idx]
        best_r = 0
        best_metric = -1
        
        if p <= WORK_L:
            for r in range(p):
                start_x = r if r >= 1 else r + p
                if start_x > WORK_L:
                    continue
                # Count how many positions with count < 2 would be covered
                positions = range(start_x, WORK_L + 1, p)
                gain = 0
                for x in positions:
                    if counts[x] < 2:
                        gain += 1
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
        else:
            # p > WORK_L: only one integer per residue
            best_r = 0
            best_metric = -1
            for r in range(1, WORK_L + 1):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
            if best_metric <= 0:
                best_r = 0
        
        residues[idx] = best_r
        
        # Update counts
        start_x = best_r if best_r >= 1 else best_r + p
        for x in range(start_x, WORK_L + 1, p):
            counts[x] += 1
    
    # Get initial score
    current_score = _find_prefix_length(counts, WORK_L)
    
    # Phase 2: Simulated Annealing with numpy-accelerated evaluation
    T_initial = 20.0
    T_final = 0.05
    cooling_rate = 0.9995
    
    T = T_initial
    best_residues = residues[:]
    best_score = current_score
    
    max_iterations = 2000000
    iteration = 0
    no_improve_count = 0
    no_improve_limit = 50000
    
    while time.time() < deadline - 15.0 and iteration < max_iterations:
        iteration += 1
        
        # Choose a prime to modify - bias toward small primes (more impact)
        # Use weighted random selection
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        
        # Choose new residue
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Compute delta score efficiently
        # Determine which positions are affected
        # Positions where x ≡ old_r (mod p) lose 1 count
        # Positions where x ≡ new_r (mod p) gain 1 count
        
        # Check if the move breaks the current prefix
        # A position x in prefix breaks if its count goes from 2 to 1
        # This happens for x ≡ old_r (mod p) where counts[x] == 2
        
        # Quick check: does old_r cover any position in [1, current_score] with count exactly 2?
        loss_in_prefix = 0
        start_old = old_r if old_r >= 1 else old_r + p
        if start_old <= current_score:
            for x in range(start_old, current_score + 1, p):
                if counts[x] == 2:
                    loss_in_prefix += 1
                    break  # One loss is enough to reject
        
        if loss_in_prefix > 0:
            # This move would break the prefix - only accept with low probability at high T
            if T < 1.0 or random.random() > 0.01:
                no_improve_count += 1
                T *= cooling_rate
                if T < T_final:
                    T = T_final
                continue
        
        # No loss in prefix - apply the move and check for extension
        # Temporarily apply
        start_old = old_r if old_r >= 1 else old_r + p
        start_new = new_r if new_r >= 1 else new_r + p
        
        # Apply changes to counts
        for x in range(start_old, WORK_L + 1, p):
            counts[x] -= 1
        for x in range(start_new, WORK_L + 1, p):
            counts[x] += 1
        
        # Find new prefix length
        new_score = current_score
        # Check if we can extend beyond current_score
        x = current_score + 1
        while x <= WORK_L and counts[x] >= 2:
            new_score = x
            x += 1
        
        # Check if prefix shrank (shouldn't happen if no loss, but verify)
        if new_score < current_score:
            # Revert
            for x in range(start_old, WORK_L + 1, p):
                counts[x] += 1
            for x in range(start_new, WORK_L + 1, p):
                counts[x] -= 1
            no_improve_count += 1
            T *= cooling_rate
            if T < T_final:
                T = T_final
            continue
        
        # Accept the move
        residues[idx] = new_r
        current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset temperature if stuck for too long
        if no_improve_count > no_improve_limit:
            T = min(T * 2.0, T_initial)
            no_improve_count = 0
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    # Use best found
    residues = best_residues
    
    # Phase 3: Targeted gap repair
    counts = _build_counts_np(residues, primes, WORK_L)
    current_score = _find_prefix_length(counts, WORK_L)
    
    improved = True
    repair_iterations = 0
    while improved and time.time() < deadline - 10.0 and repair_iterations < 100000:
        improved = False
        repair_iterations += 1
        
        # Find the first gap
        if current_score >= WORK_L:
            break
        
        gap_pos = current_score + 1
        if gap_pos > WORK_L:
            break
        
        # Find which primes cover gap_pos
        covering = []
        for i, (p, r) in enumerate(zip(primes, residues)):
            if gap_pos % p == r % p:
                covering.append(i)
        
        # We need at least 2 covering. If we have 0 or 1, try to add more
        if len(covering) < 2:
            # Try to change a non-covering prime to cover gap_pos
            # Prioritize small primes (more flexibility)
            candidates = [i for i in range(n_primes) if i not in covering]
            candidates.sort(key=lambda i: primes[i])
            
            for i in candidates:
                p = primes[i]
                old_r = residues[i]
                # Residue needed: gap_pos mod p
                new_r = gap_pos % p
                if new_r == old_r:
                    continue
                
                # Check if this move helps without breaking the prefix
                loss = 0
                start_old = old_r if old_r >= 1 else old_r + p
                if start_old <= current_score:
                    for x in range(start_old, current_score + 1, p):
                        if counts[x] == 2:
                            loss += 1
                            break
                
                if loss == 0:
                    # Apply
                    for x in range(start_old, WORK_L + 1, p):
                        counts[x] -= 1
                    start_new = new_r if new_r >= 1 else new_r + p
                    for x in range(start_new, WORK_L + 1, p):
                        counts[x] += 1
                    residues[i] = new_r
                    improved = True
                    break
        
        # Recompute score
        if improved:
            current_score = _find_prefix_length(counts, WORK_L)
    
    # Phase 4: Final polish - try to extend further with more aggressive search
    if time.time() < deadline - 5.0:
        counts = _build_counts_np(residues, primes, WORK_L)
        current_score = _find_prefix_length(counts, WORK_L)
        
        # Extended SA with focus on extending the prefix
        T = 5.0
        polish_iters = 0
        while time.time() < deadline - 3.0 and polish_iters < 100000:
            polish_iters += 1
            
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Check for loss in prefix
            loss_in_prefix = 0
            start_old = old_r if old_r >= 1 else old_r + p
            if start_old <= current_score:
                for x in range(start_old, current_score + 1, p):
                    if counts[x] == 2:
                        loss_in_prefix += 1
                        break
            
            if loss_in_prefix > 0:
                continue
            
            # Apply and check extension
            start_new = new_r if new_r >= 1 else new_r + p
            
            for x in range(start_old, WORK_L + 1, p):
                counts[x] -= 1
            for x in range(start_new, WORK_L + 1, p):
                counts[x] += 1
            
            new_score = current_score
            x = current_score + 1
            while x <= WORK_L and counts[x] >= 2:
                new_score = x
                x += 1
            
            if new_score >= current_score:
                residues[idx] = new_r
                current_score = new_score
            else:
                # Revert
                for x in range(start_old, WORK_L + 1, p):
                    counts[x] += 1
                for x in range(start_new, WORK_L + 1, p):
                    counts[x] -= 1
            
            T *= 0.999
    
    # Ensure all residues are proper Python ints
    residues = [int(r) % p for r, p in zip(residues, primes)]
    
    return residues
# EVOLVE-BLOCK-END
