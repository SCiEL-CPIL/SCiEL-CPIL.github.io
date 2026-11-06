import numpy as np
import time
import random

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    N = 50000  # Fixed working window for SA
    
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts
    
    def get_score(counts, N):
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1
    
    def count_vulnerable(counts, N):
        """Count integers with coverage exactly 1 (vulnerable)."""
        return int(np.sum(counts[1:N+1] == 1))
    
    def apply_move(counts, residues, i, r_new, N):
        p = primes[i]
        r_old = int(residues[i])
        if r_old == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_old <= N:
                counts[r_old::p] -= 1
                
        if r_new == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_new <= N:
                counts[r_new::p] += 1
        residues[i] = r_new
    
    def revert_move(counts, residues, i, r_old, N):
        p = primes[i]
        r_new = int(residues[i])
        if r_new == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_new <= N:
                counts[r_new::p] -= 1
                
        if r_old == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_old <= N:
                counts[r_old::p] += 1
        residues[i] = r_old
    
    # Vectorized simulated annealing optimizing vulnerable count
    def simulated_annealing(residues_init, N, max_steps, initial_temp, cooling):
        """SA that minimizes vulnerable count (coverage==1) as primary objective."""
        residues = residues_init.copy()
        counts = build_counts(residues, N)
        cur_vuln = count_vulnerable(counts, N)
        cur_score = get_score(counts, N)
        
        best_residues = residues.copy()
        best_vuln = cur_vuln
        best_score = cur_score
        
        temp = initial_temp
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random prime to change
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            r_new = int(rng.integers(0, p))
            
            if r_new == r_old:
                continue
            
            # Apply move
            apply_move(counts, residues, i, r_new, N)
            new_vuln = count_vulnerable(counts, N)
            new_score = get_score(counts, N)
            
            # Acceptance: prefer lower vulnerable, accept worse with probability
            delta_vuln = new_vuln - cur_vuln
            delta_score = new_score - cur_score
            
            if delta_vuln < 0 or (delta_vuln == 0 and delta_score >= 0):
                accept = True
            else:
                # Accept with Boltzmann probability based on vulnerable change
                # Scale: each vulnerable unit is "energy"
                accept_prob = np.exp(-delta_vuln / (temp + 1e-9))
                accept = rng.random() < accept_prob
            
            if accept:
                cur_vuln = new_vuln
                cur_score = new_score
                if cur_vuln < best_vuln or (cur_vuln == best_vuln and cur_score > best_score):
                    best_vuln = cur_vuln
                    best_score = cur_score
                    best_residues = residues.copy()
            else:
                revert_move(counts, residues, i, r_old, N)
            
            temp *= cooling
        
        return best_residues, best_vuln, best_score
    
    # Multi-restart SA with different initializations
    best_residues = None
    best_global_vuln = float('inf')
    best_global_score = 0
    
    # Several initial strategies
    initial_strategies = [
        lambda: np.array([rng.integers(0, p) for p in primes], dtype=np.int64),
        lambda: np.zeros(n_primes, dtype=np.int64),
        lambda: np.array([p // 2 for p in primes], dtype=np.int64),
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([int(rng.integers(0, p)) for p in primes], dtype=np.int64),
        lambda: np.array([int(rng.integers(0, p)) for p in primes], dtype=np.int64),
        lambda: np.array([int(rng.integers(0, p)) for p in primes], dtype=np.int64),
    ]
    
    sa_configs = [
        # (max_steps, initial_temp, cooling)
        (20000, 50.0, 0.995),
        (15000, 30.0, 0.997),
        (10000, 20.0, 0.998),
        (8000, 10.0, 0.999),
    ]
    
    restart_count = 0
    for strat_idx, strat in enumerate(initial_strategies):
        if time.time() > deadline - 10.0:
            break
        
        for sa_idx, (max_steps, init_temp, cool) in enumerate(sa_configs):
            if time.time() > deadline - 10.0:
                break
            
            residues_init = strat()
            best_res, best_v, best_s = simulated_annealing(residues_init, N, max_steps, init_temp, cool)
            
            if best_v < best_global_vuln or (best_v == best_global_vuln and best_s > best_global_score):
                best_global_vuln = best_v
                best_global_score = best_s
                best_residues = best_res.copy()
            
            restart_count += 1
    
    # If we have time, do a final polish with focused SA around the best solution
    if best_residues is not None and time.time() < deadline - 5.0:
        # Polish: run SA with very low temperature from best solution
        remaining_steps = int((deadline - time.time() - 2.0) * 3000)  # ~3000 steps/sec estimate
        remaining_steps = max(remaining_steps, 1000)
        remaining_steps = min(remaining_steps, 50000)
        
        polished_res, polished_v, polished_s = simulated_annealing(
            best_residues, N, remaining_steps, 5.0, 0.9995
        )
        
        if polished_v < best_global_vuln or (polished_v == best_global_vuln and polished_s > best_global_score):
            best_global_vuln = polished_v
            best_global_score = polished_s
            best_residues = polished_res.copy()
    
    # Verify final score
    if best_residues is not None:
        counts_final = build_counts(best_residues, N)
        final_score = get_score(counts_final, N)
        if final_score > best_global_score:
            best_global_score = final_score
    
    return best_residues.tolist()
