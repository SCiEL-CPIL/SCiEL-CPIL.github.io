import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (evaluator kills at 150s)
    
    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 50000
    
    best_L_found = 0
    best_residues = None
    
    def compute_L(cov):
        """Find the largest L such that cov[1..L] >= 2"""
        mask = cov[1:] < 2
        if not mask.any():
            return W
        first_bad_idx = np.argmax(mask)
        return first_bad_idx
    
    def apply_move(cov, residues, i, new_r):
        """Apply a single move: change residues[i] to new_r. Returns old_r."""
        p = primes_arr[i]
        old_r = residues[i]
        if old_r == new_r:
            return old_r
        old_start = old_r if old_r > 0 else p
        new_start = new_r if new_r > 0 else p
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        if new_start <= W:
            cov[new_start:W+1:p] += 1
        residues[i] = new_r
        return old_r
    
    def revert_move(cov, residues, i, new_r, old_r):
        """Revert a move: change residues[i] from new_r back to old_r."""
        p = primes_arr[i]
        if new_r == old_r:
            return
        old_start = new_r if new_r > 0 else p
        new_start = old_r if old_r > 0 else p
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        if new_start <= W:
            cov[new_start:W+1:p] += 1
        residues[i] = old_r
    
    def estimate_damage(cov, residues, i, new_r, bad_pos):
        """Estimate damage of changing prime i from old_r to new_r.
        Focus on positions in [1, bad_pos-1] that might lose coverage."""
        p = primes_arr[i]
        old_r = residues[i]
        if old_r == new_r:
            return 0
        
        # Positions that will lose coverage: x in [1, bad_pos-1] where x % p == old_r but x % p != new_r
        # And gain: x in [1, bad_pos-1] where x % p == new_r but x % p != old_r
        
        # For efficiency, check positions where coverage is exactly 2 (critical)
        # Old positions: old_r, old_r+p, ... < bad_pos
        # New positions: new_r, new_r+p, ... < bad_pos
        
        damage = 0
        
        # Check old positions that lose coverage
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start < bad_pos:
            # Vectorized check for old positions
            old_positions = np.arange(old_start, bad_pos, p)
            # These positions lose this prime's coverage
            # Check how many have cov == 2 (they'll drop to 1)
            if len(old_positions) > 0:
                old_cov = cov[old_positions]
                damage += np.sum(old_cov == 2)
        
        # Check new positions that gain coverage
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start < bad_pos:
            new_positions = np.arange(new_start, bad_pos, p)
            if len(new_positions) > 0:
                new_cov = cov[new_positions]
                # Gaining coverage where cov == 1 helps (reduces effective damage)
                damage -= np.sum(new_cov == 1)
        
        return damage
    
    def greedy_init():
        """Greedy initialization: for each prime, choose r_p to maximize coverage of small numbers."""
        residues = np.zeros(n_primes, dtype=np.int64)
        cov = np.zeros(W + 1, dtype=np.int32)
        
        for i in range(n_primes):
            p = primes_arr[i]
            best_r = 0
            best_score = -float('inf')
            
            # Try a few candidate residues
            candidates = [0, 1, 2, 3, 4]
            if p > 5:
                candidates.extend([p//2, p//3, 2*p//3, random.randint(0, p-1)])
            
            for r in candidates:
                if r >= p:
                    continue
                # Estimate coverage contribution in [1, min(200, W)]
                check_range = min(200, W)
                start = r if r > 0 else p
                if start <= check_range:
                    count = (check_range - start) // p + 1
                else:
                    count = 0
                score = count
                if score > best_score:
                    best_score = score
                    best_r = r
            
            residues[i] = best_r
            start = best_r if best_r > 0 else p
            if start <= W:
                cov[start:W+1:p] += 1
        
        return residues, cov
    
    def random_init():
        """Random initialization."""
        residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int64)
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r > 0 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return residues, cov
    
    def simulated_annealing(residues, cov, current_L, max_steps=5000):
        """Simulated annealing with single-residue moves."""
        local_best_L = current_L
        local_best_res = residues.copy()
        
        T = 50.0
        T_min = 0.1
        cooling = 0.998
        
        stuck_counter = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if current_L >= W:
                break
            
            # Choose a random prime to modify
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            
            # Choose new residue: biased toward fixing the first gap
            if random.random() < 0.3 and current_L < W:
                # Target the first bad position
                bad_pos = current_L + 1
                new_r = bad_pos % p
            else:
                new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply move
            apply_move(cov, residues, i, new_r)
            new_L = compute_L(cov)
            
            # Compute delta in objective
            delta_L = new_L - current_L
            
            # Accept if improvement, or with probability exp(-|delta|/T) for worsening
            if delta_L >= 0:
                accept = True
            else:
                # Only accept worsening moves if they don't hurt too much
                # Use a tempered criterion
                accept = random.random() < np.exp(delta_L / max(T, 0.01))
            
            if accept:
                current_L = new_L
                if new_L > local_best_L:
                    local_best_L = new_L
                    local_best_res = residues.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                # Revert
                revert_move(cov, residues, i, new_r, old_r)
                stuck_counter += 1
            
            T = max(T * cooling, T_min)
            
            # If stuck for too long, do a random kick
            if stuck_counter > 200:
                kick_i = random.randint(0, n_primes - 1)
                kick_p = primes_arr[kick_i]
                kick_new_r = random.randint(0, kick_p - 1)
                kick_old_r = residues[kick_i]
                if kick_new_r != kick_old_r:
                    apply_move(cov, residues, kick_i, kick_new_r)
                    new_L = compute_L(cov)
                    current_L = new_L
                    if new_L > local_best_L:
                        local_best_L = new_L
                        local_best_res = residues.copy()
                    stuck_counter = 0
            
            # Periodically try to fix the first gap directly
            if step % 50 == 0 and current_L < W:
                bad_pos = current_L + 1
                # Find a prime that doesn't currently cover bad_pos
                candidates = []
                for j in range(n_primes):
                    pj = primes_arr[j]
                    if bad_pos % pj != residues[j]:
                        candidates.append(j)
                if candidates:
                    j = random.choice(candidates)
                    new_r = bad_pos % primes_arr[j]
                    old_r = residues[j]
                    if new_r != old_r:
                        apply_move(cov, residues, j, new_r)
                        new_L = compute_L(cov)
                        if new_L > current_L:
                            current_L = new_L
                            if new_L > local_best_L:
                                local_best_L = new_L
                                local_best_res = residues.copy()
                        else:
                            revert_move(cov, residues, j, new_r, old_r)
        
        return local_best_L, local_best_res
    
    # Main Loop: Multiple Restarts
    restarts = 0
    use_greedy = True
    
    while time.time() < deadline - 10:
        # Alternate between greedy and random init
        if use_greedy:
            residues, cov = greedy_init()
        else:
            residues, cov = random_init()
        use_greedy = not use_greedy
        
        current_L = compute_L(cov)
        
        L, res = simulated_annealing(residues, cov, current_L, max_steps=8000)
        
        if L > best_L_found:
            best_L_found = L
            best_residues = res.copy()
        
        restarts += 1
        # print(f"Restart {restarts}: L={L}", file=sys.stderr)
    
    if best_residues is None:
        # Fallback: all 1s
        return [1] * n_primes
        
    return [int(r) for r in best_residues]
