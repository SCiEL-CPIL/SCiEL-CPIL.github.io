import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We evaluate up to a large limit to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    MAX_EVAL = 20000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def get_coverage(residues):
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    # Initial solution: Random residues
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Strategy: 
    # Instead of just fixing the first gap, we use a "Matching-Guided Repair".
    # We identify the set of "deficient" integers in the current prefix (coverage < 2).
    # For each deficient integer x, we know which primes cover it.
    # We want to assign residues to primes such that the "deficit" is minimized.
    # 
    # A key insight: If we fix a target L, we can check if there exists a set of primes 
    # that can "support" the coverage. 
    # However, the constraint is "at least 2" per integer. This is a b-matching problem.
    # 
    # Let's try a different angle: 
    # The current greedy approach gets stuck when no single prime change fixes the next integer.
    # This happens when the next integer x needs 2 more primes to cover it, but changing any 
    # prime to cover x breaks coverage for some previous integer y.
    # 
    # We can model this as: 
    # Let S be the set of integers in 1..L. 
    # For each prime p, it covers a subset C_p of S.
    # We want to choose residues such that |{p : x in C_p}| >= 2 for all x in S.
    # 
    # If we are stuck at L, it means x = L+1 has coverage < 2.
    # Let k be the number of primes currently covering x. We need 2-k more.
    # We can try to "swap" residues. 
    # 
    # New Strategy: 
    # 1. Compute coverage for 1..MAX_EVAL.
    # 2. Find the first failure x_fail.
    # 3. Identify primes p that do NOT cover x_fail.
    # 4. For each such prime p, changing its residue to x_fail % p will increase coverage of x_fail by 1.
    #    But it will remove p from the set of primes covering some other integers.
    #    Specifically, p used to cover {y : y = r_p (mod p)}. Now it covers {y : y = x_fail % p (mod p)}.
    #    The "lost" coverage is for integers y in the old set that are not in the new set.
    #    The "gained" coverage is for integers y in the new set that are not in the old set.
    #
    # We can estimate the "damage" of a change by counting how many integers in 1..L 
    # that were covered by p (and are now not) have coverage exactly 2. 
    # If any such integer exists, its coverage drops to 1, creating a new failure.
    #
    # Let's implement a "Lookahead Repair" that checks the impact on the entire prefix.
    # To make this faster, we precompute for each prime p and each integer x in 1..MAX_EVAL 
    # whether p covers x.
    # But this is just the hits matrix.
    
    # Let's try a Simulated Annealing approach with a different objective function:
    # Minimize the "deficit" = sum( max(0, 2 - coverage[x]) for x in 1..MAX_EVAL )
    # This allows us to make moves that might temporarily lower the prefix length 
    # but reduce the total deficit, potentially escaping local minima.
    
    def calculate_deficit(residues):
        coverage = get_coverage(residues)
        # Deficit is only relevant for the prefix? 
        # No, for the score, only the prefix matters. 
        # But for optimization, a global deficit might guide better.
        # Let's focus on the prefix up to best_score + some margin.
        # Actually, let's just use the prefix up to MAX_EVAL.
        deficit = np.sum(np.maximum(0, 2 - coverage))
        return deficit

    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    current_deficit = calculate_deficit(residues)
    best_deficit = current_deficit
    best_residues_sa = residues.copy()
    
    # We will alternate between SA and Greedy Repair
    # SA to escape local minima, Greedy to extend the prefix.
    
    while time.time() < deadline:
        # Phase 1: Greedy Repair (fast, extends prefix)
        current_score = evaluate(residues)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        if current_score >= MAX_EVAL:
            break
            
        target_x = current_score + 1
        
        # Find candidates for repair
        candidates = [i for i, p in enumerate(primes) if (target_x - residues[i]) % p != 0]
        if not candidates:
            break
            
        random.shuffle(candidates)
        found_repair = False
        for idx in candidates:
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            if old_res == new_res: continue
            
            residues[idx] = new_res
            new_score = evaluate(residues)
            if new_score >= current_score:
                found_repair = True
                break
            else:
                residues[idx] = old_res
        
        if found_repair:
            continue
            
        # Phase 2: SA Kick (escape local minimum)
        # Try a few random moves
        for _ in range(10):
            if time.time() > deadline: break
            
            # Pick a random prime to change
            idx = random.randint(0, n_primes - 1)
            old_res = residues[idx]
            new_res = random.randint(0, primes[idx] - 1)
            if new_res == old_res: continue
            
            residues[idx] = new_res
            new_deficit = calculate_deficit(residues)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_deficit - current_deficit
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_deficit = new_deficit
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    best_residues_sa = residues.copy()
            else:
                residues[idx] = old_res
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Reset or restart?
            # Let's just keep going.
            
        # Periodically, if SA finds a better deficit, check if it leads to a better score
        # and adopt it if so.
        if best_deficit < current_deficit:
            # Adopt the best found so far in SA
            # But only if it doesn't break the current prefix too badly?
            # Actually, let's just check the score of best_residues_sa
            sa_score = evaluate(best_residues_sa)
            if sa_score > current_score:
                residues = best_residues_sa.copy()
                current_score = sa_score
                current_deficit = calculate_deficit(residues)
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
    return best_residues.tolist()
