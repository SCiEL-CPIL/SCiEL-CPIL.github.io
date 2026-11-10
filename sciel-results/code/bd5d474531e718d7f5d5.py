import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial parameters
    M = 2000
    max_M = 100000
    
    # Initial residues: try to cover early numbers
    residues = [0] * n_primes
    
    def recompute_counts(residues, M, primes):
        counts = np.zeros(M + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[r:M+1:p] += 1
        return counts

    def get_L(counts, M):
        uncovered = np.where(counts[1:M+1] < 2)[0]
        if len(uncovered) > 0:
            return int(uncovered[0])
        return M

    # Greedy first-bad-point initialization
    residues = [0] * n_primes
    counts = recompute_counts(residues, M, primes)
    
    # Greedy repair: find first bad point, try to fix it
    for _ in range(200):  # Limited iterations for initialization
        L = get_L(counts, M)
        if L >= M:
            break
        bad_x = L + 1  # First uncovered point
        
        # Find primes that currently cover bad_x
        covering = []
        for i, p in enumerate(primes):
            r = residues[i]
            if bad_x % p == (r % p):
                covering.append(i)
        
        # If fewer than 2 primes cover bad_x, try to reassign one
        if len(covering) < 2:
            # Try to find a prime to reassign to cover bad_x
            # Prefer primes that don't currently cover bad_x
            candidates = [i for i in range(n_primes) if i not in covering]
            if not candidates:
                candidates = list(range(n_primes))
            
            random.shuffle(candidates)
            improved = False
            for idx in candidates[:20]:  # Try up to 20 candidates
                p = primes[idx]
                old_r = residues[idx]
                new_r = bad_x % p
                
                if new_r == old_r:
                    continue
                
                # Apply change
                if old_r == 0:
                    counts[p:M+1:p] -= 1
                else:
                    counts[old_r:M+1:p] -= 1
                if new_r == 0:
                    counts[p:M+1:p] += 1
                else:
                    counts[new_r:M+1:p] += 1
                residues[idx] = new_r
                
                new_L = get_L(counts, M)
                
                if new_L >= L:
                    improved = True
                    break
                else:
                    # Revert
                    residues[idx] = old_r
                    if new_r == 0:
                        counts[p:M+1:p] -= 1
                    else:
                        counts[new_r:M+1:p] -= 1
                    if old_r == 0:
                        counts[p:M+1:p] += 1
                    else:
                        counts[old_r:M+1:p] += 1
            
            if not improved:
                break
    
    best_L = get_L(counts, M)
    best_residues = residues.copy()
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.999
    
    # Track stagnation
    last_improvement_time = start_time
    stagnation_limit = 3.0
    
    # Helper: targeted repair - focus on first bad point
    def targeted_repair(residues, counts, M, primes, n_primes, max_tries=50):
        """Try to improve by fixing the first uncovered point using heuristic cost."""
        L = get_L(counts, M)
        if L >= M:
            return L, residues, counts
        
        bad_x = L + 1
        covering = [i for i, p in enumerate(primes) if bad_x % p == (residues[i] % p)]
        
        if len(covering) >= 2:
            return L, residues, counts
        
        # Try to reassign primes to cover bad_x
        candidates = [i for i in range(n_primes) if i not in covering]
        if not candidates:
            candidates = list(range(n_primes))
        
        # Heuristic: Sort candidates by "damage" to the current prefix [1, L]
        # Damage = number of points y in [1, L] such that y % p == old_r and counts[y] == 2
        scored_candidates = []
        for idx in candidates:
            p = primes[idx]
            old_r = residues[idx]
            # Calculate damage
            # Points are old_r, old_r+p, old_r+2p, ... <= L
            # We need to count how many have counts[x] == 2
            # Vectorized check
            points = np.arange(old_r if old_r > 0 else p, L + 1, p)
            if len(points) > 0:
                damage = np.sum(counts[points] == 2)
            else:
                damage = 0
            scored_candidates.append((damage, idx))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        for damage, idx in scored_candidates[:max_tries]:
            p = primes[idx]
            old_r = residues[idx]
            new_r = bad_x % p
            
            if new_r == old_r:
                continue
            
            # Apply
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            residues[idx] = new_r
            
            new_L = get_L(counts, M)
            
            if new_L > L:
                return new_L, residues, counts
            else:
                # Revert
                residues[idx] = old_r
                if new_r == 0:
                    counts[p:M+1:p] -= 1
                else:
                    counts[new_r:M+1:p] -= 1
                if old_r == 0:
                    counts[p:M+1:p] += 1
                else:
                    counts[old_r:M+1:p] += 1
        
        return L, residues, counts
    
    # Main SA loop
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
        
        # Alternate between random moves and targeted repairs
        if random.random() < 0.3:
            # Targeted repair
            L, residues, counts = targeted_repair(residues, counts, M, primes, n_primes, max_tries=30)
            if L > best_L:
                best_L = L
                best_residues = residues.copy()
                last_improvement_time = time.time()
                
                if best_L == M:
                    old_M = M
                    M = min(M * 2, max_M)
                    counts = recompute_counts(residues, M, primes)
                    best_L = get_L(counts, M)
        else:
            # Random move
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            residues[idx] = new_r
            
            new_L = get_L(counts, M)
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                last_improvement_time = time.time()
                
                if best_L == M:
                    old_M = M
                    M = min(M * 2, max_M)
                    counts = recompute_counts(residues, M, primes)
                    best_L = get_L(counts, M)
            else:
                delta = new_L - best_L
                if delta < 0:
                    prob = np.exp(delta / T)
                    if random.random() >= prob:
                        # Revert
                        residues[idx] = old_r
                        if new_r == 0:
                            counts[p:M+1:p] -= 1
                        else:
                            counts[new_r:M+1:p] -= 1
                        if old_r == 0:
                            counts[p:M+1:p] += 1
                        else:
                            counts[old_r:M+1:p] += 1
        
        T *= cooling
        
        # Stagnation check
        if time.time() - last_improvement_time > stagnation_limit:
            # Restart from best with perturbation
            residues = best_residues.copy()
            counts = recompute_counts(residues, M, primes)
            best_L = get_L(counts, M)
            
            # Perturb
            n_perturb = random.randint(3, 8)
            for _ in range(n_perturb):
                j = random.randint(0, n_primes - 1)
                p_j = primes[j]
                old_r = residues[j]
                new_r = random.randint(0, p_j - 1)
                if new_r == old_r:
                    continue
                residues[j] = new_r
                if old_r == 0:
                    counts[p_j:M+1:p_j] -= 1
                else:
                    counts[old_r:M+1:p_j] -= 1
                if new_r == 0:
                    counts[p_j:M+1:p_j] += 1
                else:
                    counts[new_r:M+1:p_j] += 1
            
            T = 50.0
            last_improvement_time = time.time()
            
    # Final Polish Phase
    # Use the best residues found so far
    residues = best_residues.copy()
    # Expand M to ensure we capture any further progress
    # Check current L
    L = get_L(recompute_counts(residues, M, primes), M)
    if L >= M:
        M = min(M * 2, max_M)
    
    counts = recompute_counts(residues, M, primes)
    L = get_L(counts, M)
    
    polish_deadline = start_time + 118
    while time.time() < polish_deadline:
        L = get_L(counts, M)
        if L >= M:
            if M < max_M:
                M = min(M * 2, max_M)
                counts = recompute_counts(residues, M, primes)
                L = get_L(counts, M)
            else:
                break
        if L >= M:
            continue
            
        bad_x = L + 1
        covering = [i for i, p in enumerate(primes) if bad_x % p == (residues[i] % p)]
        
        if len(covering) >= 2:
            continue
            
        candidates = [i for i in range(n_primes) if i not in covering]
        if not candidates:
            candidates = list(range(n_primes))
            
        scored_candidates = []
        for idx in candidates:
            p = primes[idx]
            old_r = residues[idx]
            points = np.arange(old_r if old_r > 0 else p, L + 1, p)
            if len(points) > 0:
                damage = np.sum(counts[points] == 2)
            else:
                damage = 0
            scored_candidates.append((damage, idx))
            
        scored_candidates.sort(key=lambda x: x[0])
        
        improved = False
        for damage, idx in scored_candidates:
            if damage > 0:
                # If best option has damage, we might be stuck. 
                # Try it only if it's the best and we have no other choice, or if we accept a temporary dip?
                # For polish, we usually want zero damage moves first.
                # If no zero damage move, we break and try a random kick?
                # Let's just try the best one (min damage) regardless, but only accept if L improves.
                pass
            
            p = primes[idx]
            old_r = residues[idx]
            new_r = bad_x % p
            
            if new_r == old_r:
                continue
            
            # Apply
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            residues[idx] = new_r
            
            new_L = get_L(counts, M)
            
            if new_L > L:
                L = new_L
                improved = True
                break
            else:
                # Revert
                residues[idx] = old_r
                if new_r == 0:
                    counts[p:M+1:p] -= 1
                else:
                    counts[new_r:M+1:p] -= 1
                if old_r == 0:
                    counts[p:M+1:p] += 1
                else:
                    counts[old_r:M+1:p] += 1
        
        if not improved:
            # Stuck. Do a small random perturbation to escape local minimum
            # Pick a random prime and change its residue
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            residues[idx] = new_r
            # Update best if this perturbation accidentally improved L?
            # Unlikely, but check
            new_L = get_L(counts, M)
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
            # Continue polish loop to try and fix the bad point created by perturbation
            # Or just continue the loop which will re-evaluate L
            
    # Final update of best
    final_L = get_L(recompute_counts(residues, M, primes), M)
    if final_L > best_L:
        best_L = final_L
        best_residues = residues.copy()
        
    return best_residues
# EVOLVE-BLOCK-END
