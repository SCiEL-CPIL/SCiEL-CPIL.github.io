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

def _find_first_gap(counts, max_L):
    """Find first index where count < 2."""
    for x in range(1, max_L + 1):
        if counts[x] < 2:
            return x
    return max_L + 1

def _count_deficit(counts, max_L):
    """Count positions with deficit (count < 2) in 1..max_L."""
    deficit = 0
    for x in range(1, max_L + 1):
        if counts[x] < 2:
            deficit += 1
    return deficit

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Working range for optimization
    WORK_L = 5000
    
    def _greedy_init(seed):
        """Greedy initialization with given seed."""
        rng = random.Random(seed)
        res = [0] * n_primes
        cnt = np.zeros(WORK_L + 1, dtype=np.int32)
        
        order = list(range(n_primes))
        rng.shuffle(order)
        
        for idx in order:
            p = primes[idx]
            best_r = 0
            best_metric = -1
            
            if p <= WORK_L:
                for r in range(p):
                    start_x = r if r >= 1 else r + p
                    if start_x > WORK_L:
                        continue
                    gain = 0
                    for x in range(start_x, WORK_L + 1, p):
                        if cnt[x] < 2:
                            gain += 1
                    if gain > best_metric:
                        best_metric = gain
                        best_r = r
            else:
                best_r = rng.randint(0, p - 1)
            
            res[idx] = best_r
            start_x = best_r if best_r >= 1 else best_r + p
            for x in range(start_x, WORK_L + 1, p):
                cnt[x] += 1
        
        return res, cnt
    
    def _sa_search(residues, counts, current_score, time_limit, T_init=20.0):
        """Simulated annealing search."""
        T = T_init
        best_res = residues[:]
        best_score = current_score
        no_improve = 0
        no_improve_limit = 30000
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Check loss in prefix
            loss = 0
            start_old = old_r if old_r >= 1 else old_r + p
            if start_old <= current_score:
                for x in range(start_old, current_score + 1, p):
                    if counts[x] == 2:
                        loss += 1
                        break
            
            if loss > 0:
                if T < 1.0 or random.random() > 0.01:
                    no_improve += 1
                    T *= 0.9995
                    if T < 0.05:
                        T = 0.05
                    continue
            else:
                # Apply move
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
                    if current_score > best_score:
                        best_score = current_score
                        best_res = residues[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    for x in range(start_old, WORK_L + 1, p):
                        counts[x] += 1
                    for x in range(start_new, WORK_L + 1, p):
                        counts[x] -= 1
                    no_improve += 1
            
            if no_improve > no_improve_limit:
                T = min(T * 2.0, T_init)
                no_improve = 0
            
            T *= 0.9995
            if T < 0.05:
                T = 0.05
        
        return best_res, best_score
    
    def _gap_repair(residues, counts, current_score, time_limit):
        """Targeted gap repair."""
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            
            if current_score >= WORK_L:
                break
            
            gap_pos = current_score + 1
            if gap_pos > WORK_L:
                break
            
            covering = []
            for i, (p, r) in enumerate(zip(primes, residues)):
                if gap_pos % p == r % p:
                    covering.append(i)
            
            if len(covering) < 2:
                candidates = [i for i in range(n_primes) if i not in covering]
                candidates.sort(key=lambda i: primes[i])
                
                for i in candidates:
                    p = primes[i]
                    old_r = residues[i]
                    new_r = gap_pos % p
                    if new_r == old_r:
                        continue
                    
                    loss = 0
                    start_old = old_r if old_r >= 1 else old_r + p
                    if start_old <= current_score:
                        for x in range(start_old, current_score + 1, p):
                            if counts[x] == 2:
                                loss += 1
                                break
                    
                    if loss == 0:
                        for x in range(start_old, WORK_L + 1, p):
                            counts[x] -= 1
                        start_new = new_r if new_r >= 1 else new_r + p
                        for x in range(start_new, WORK_L + 1, p):
                            counts[x] += 1
                        residues[i] = new_r
                        improved = True
                        break
            
            if improved:
                current_score = _find_prefix_length(counts, WORK_L)
        
        return residues, current_score
    
    # Multi-restart strategy
    best_overall_res = None
    best_overall_score = 0
    
    # Time allocation: 3 restarts with greedy init + SA + repair
    num_restarts = 3
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        seed = restart * 12345 + 42
        residues, counts = _greedy_init(seed)
        current_score = _find_prefix_length(counts, WORK_L)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_res = residues[:]
        
        # SA search
        sa_time = time_per_restart * 0.6
        residues, current_score = _sa_search(residues, counts, current_score, sa_time)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_res = residues[:]
        
        # Gap repair
        counts = _build_counts_np(residues, primes, WORK_L)
        current_score = _find_prefix_length(counts, WORK_L)
        
        repair_time = time_per_restart * 0.3
        residues, current_score = _gap_repair(residues, counts, current_score, repair_time)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_res = residues[:]
        
        # Final SA polish
        counts = _build_counts_np(residues, primes, WORK_L)
        current_score = _find_prefix_length(counts, WORK_L)
        
        polish_time = time_per_restart * 0.1
        residues, current_score = _sa_search(residues, counts, current_score, polish_time, T_init=5.0)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_res = residues[:]
    
    # Use best found
    residues = best_overall_res if best_overall_res is not None else [0] * n_primes
    
    # Final gap repair on best solution
    counts = _build_counts_np(residues, primes, WORK_L)
    current_score = _find_prefix_length(counts, WORK_L)
    
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 2.0:
        residues, current_score = _gap_repair(residues, counts, current_score, remaining_time)
    
    # Ensure all residues are proper Python ints
    residues = [int(r) % p for r, p in zip(residues, primes)]
    
    return residues
# EVOLVE-BLOCK-END
