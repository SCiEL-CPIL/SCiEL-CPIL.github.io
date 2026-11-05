import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # 2. Helper functions
    def calc_L(P, residues, max_eval):
        covered = np.zeros(max_eval + 1, dtype=bool)
        for p in P:
            r = residues.get(p, 0)
            start = r if r > 0 else p
            if start <= max_eval:
                idx = np.arange(start, max_eval + 1, p)
                covered[idx] = True
        
        # Find first gap in 1..max_eval
        # ~covered[1:] gives boolean array for 1..max_eval
        uncovered = np.where(~covered[1:max_eval+1])[0]
        if len(uncovered) == 0:
            return max_eval
        return uncovered[0]

    def greedy_residues(P, max_eval):
        covered = np.zeros(max_eval + 1, dtype=bool)
        residues = {}
        # Sort by prime size
        for p in sorted(P):
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = r if r > 0 else p
                if start > max_eval:
                    continue
                idx = np.arange(start, max_eval + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[p] = best_r
            start = best_r if best_r > 0 else p
            if start <= max_eval:
                idx = np.arange(start, max_eval + 1, p)
                covered[idx] = True
        return residues, covered

    # 3. Initialization
    # Start with a set of small primes that fits budget
    # 3,5,7,11,13,17,19,23 sum ~ 0.99
    current_P = [3, 5, 7, 11, 13, 17, 19, 23]
    max_eval = 1000
    
    current_residues, _ = greedy_residues(current_P, max_eval)
    current_L = calc_L(current_P, current_residues, max_eval)
    
    best_L = current_L
    best_classes = [[p, current_residues[p]] for p in current_P]
    
    # 4. Local Search
    no_improve_count = 0
    
    while time.time() < deadline - 10:
        # Expand evaluation window if fully covered
        if current_L >= max_eval:
            max_eval = min(max_eval * 2, 100000)
            current_residues, _ = greedy_residues(current_P, max_eval)
            current_L = calc_L(current_P, current_residues, max_eval)
            if current_L > best_L:
                best_L = current_L
                best_classes = [[p, current_residues[p]] for p in current_P]
            no_improve_count = 0
            continue
        
        improved = False
        smallest_uncovered = current_L + 1
        
        # A. Residue Optimization
        # Try to cover smallest_uncovered
        for p in current_P:
            target_r = smallest_uncovered % p
            if target_r == current_residues[p]:
                continue
                
            trial_residues = dict(current_residues)
            trial_residues[p] = target_r
            trial_L = calc_L(current_P, trial_residues, max_eval)
            
            if trial_L > current_L:
                current_residues = trial_residues
                current_L = trial_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[p, current_residues[p]] for p in current_P]
                smallest_uncovered = current_L + 1
                # Break and restart loop to re-evaluate with new smallest_uncovered
                break
        
        # B. Prime Swapping (if no residue improvement)
        if not improved:
            # Try swapping a large prime with a smaller one
            P_desc = sorted(current_P, reverse=True)
            small_primes = [p for p in primes if p not in current_P and p < 100]
            
            for p_out in P_desc[:3]:
                sum_out = sum(1.0/q for q in current_P if q != p_out)
                
                for p_in in small_primes[:3]:
                    if sum_out + 1.0/p_in > 1.0:
                        continue
                    
                    trial_P = [q for q in current_P if q != p_out] + [p_in]
                    trial_residues, _ = greedy_residues(trial_P, max_eval)
                    trial_L = calc_L(trial_P, trial_residues, max_eval)
                    
                    if trial_L > current_L:
                        current_P = trial_P
                        current_residues = trial_residues
                        current_L = trial_L
                        improved = True
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = [[p, current_residues[p]] for p in current_P]
                        break
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            # If stuck for a while, try a random perturbation
            if no_improve_count > 10:
                # Pick a random prime and change its residue randomly
                p_rand = current_P[np.random.randint(len(current_P))]
                r_rand = np.random.randint(p_rand)
                current_residues[p_rand] = r_rand
                current_L = calc_L(current_P, current_residues, max_eval)
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[p, current_residues[p]] for p in current_P]
                no_improve_count = 0

    # Final Verification
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback to a valid set
        return [[p, 0] for p in [13, 17, 19]]
        
    return best_classes
