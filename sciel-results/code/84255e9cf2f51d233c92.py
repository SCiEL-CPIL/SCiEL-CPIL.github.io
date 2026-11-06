import numpy as np
import time

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
    
    # We search for L up to 50,000. 
    # If we find a solution covering 50,000, we verify up to 100,000 at the end.
    N = 50000
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    
    def build_counts(residues):
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

    def get_score(counts):
        # Find first index >= 1 where counts < 2
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    # Initial Strategies
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
        
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
        
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
        
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]
    
    # Precompute prime indices for faster access
    prime_indices = np.arange(n_primes)
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        counts = build_counts(residues)
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Simulated Annealing Logic
        # We optimize a surrogate: minimize "vulnerable" points (count == 1) 
        # and maximize the first failure position.
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        
        stall_count = 0
        max_stall = 1000
        
        while time.time() < deadline - 5.0:
            if current_score == N:
                break
            
            x_fail = current_score
            
            # Move Selection
            # 1. Greedy Repair: Pick a prime that does NOT cover x_fail, set r = x_fail % p
            # 2. Random Walk: Pick random prime, random residue
            
            # Bias towards repair moves when T is low, random when T is high
            if rng.random() < 0.7:
                # Repair move
                # Find a prime that doesn't cover x_fail
                # We can't easily check all primes quickly without a loop, so we pick random
                # and check.
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = int(residues[i])
                
                if (x_fail % p) != r_old:
                    r_new = x_fail % p
                else:
                    # This prime covers x_fail. Changing it is risky.
                    # Let's just do a random move instead.
                    r_new = int(rng.integers(0, p))
                    if r_new == r_old:
                        continue
            else:
                # Random move
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = int(residues[i])
                r_new = int(rng.integers(0, p))
                if r_new == r_old:
                    continue
            
            # Apply move to counts
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
            
            new_score = get_score(counts)
            
            # Calculate delta in score
            delta_score = new_score - current_score
            
            # Calculate change in "vulnerable" set size (count == 1)
            # This is expensive to compute from scratch every time.
            # Instead, we can just use the score as the primary objective.
            # The "simulated annealing" aspect is accepting worse moves with probability exp(-delta/T)
            
            if delta_score > 0:
                accepted = True
            elif delta_score == 0:
                # Accept equal score moves with some probability to explore
                accepted = rng.random() < 0.1
            else:
                # Worse score. Accept with probability exp(-|delta| / T)
                # But delta can be large.
                # Let's cap the probability.
                prob = np.exp(-abs(delta_score) / (T * 100)) # Scale T
                accepted = rng.random() < prob
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert counts
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
                        
                stall_count += 1
                if stall_count > max_stall:
                    break
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1
        
        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1
            
    return best_residues.tolist()
