import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # We will try to maximize L. 
    # Strategy: 
    # 1. Define a feasibility check for a target L: Can we assign residues such that 
    #    every x in 1..L has coverage >= 2?
    # 2. Use Backtracking Search with Forward Checking.
    #    - State: residues assigned so far.
    #    - Constraint: For all x in 1..L, count of primes p_i where (x - r_i) % p_i == 0 must be >= 2.
    #    - This is equivalent to: For each x, the number of "active" primes covering x is >= 2.
    #    - However, checking all x for every step is O(L * n_primes).
    #    - Optimization: Maintain a coverage array. When assigning r_i for prime i, 
    #      update coverage[x] for all x in 1..L where x % p_i == r_i.
    
    # To make this tractable, we need a good heuristic for variable ordering.
    # Variable: Prime index i.
    # Value: Residue r_i in [0, p_i).
    
    # We will use a recursive backtracking solver that tries to satisfy constraints for a fixed L.
    # If it fails, we reduce L and try again. We start with a high L (e.g., 1000) and decrease.
    # Actually, we can just search for the maximum L directly by trying to extend the prefix.
    
    # Let's use a different approach: Local Search with a very fast, vectorized "gap-filling" objective.
    # The previous program's bottleneck was the pure Python loop in compute_score.
    # We will use NumPy to compute the coverage array in O(sum(L/p_i)) time, which is fast.
    
    MAX_RANGE = 2000 # We search up to this, but score is capped at 10^5. 
                     # If we reach 1000, we are good.
    
    def compute_coverage_array(residues, L):
        """
        Returns an array of length L+1 where arr[x] is the number of primes covering x.
        """
        cover = np.zeros(L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start_x = p
            else:
                start_x = r
            
            if start_x > L:
                continue
            
            # Vectorized increment
            # Indices: start_x, start_x + p, ... <= L
            indices = np.arange(start_x, L + 1, p)
            cover[indices] += 1
        return cover
    
    def get_score(residues, max_check=2000):
        """
        Computes the longest prefix L such that coverage >= 2.
        """
        # We check in chunks or just up to max_check
        # If L > 1000, we've likely found a good solution.
        cover = compute_coverage_array(residues, max_check)
        # Find first index where cover < 2
        # cover[0] is unused. We care about 1..max_check
        # np.where(cover[1:] < 2)[0] gives indices in 0..max_check-1 corresponding to 1..max_check
        gaps = np.where(cover[1:] < 2)[0]
        if len(gaps) == 0:
            return max_check
        return gaps[0] # This is the 0-based index in cover[1:], so the value is gaps[0]
        # Wait, if gaps[0] is 0, it means x=1 failed. Score is 0? No, score is largest L.
        # If x=1 fails, L=0.
        # If x=5 fails, L=4.
        # So L = gaps[0]
        
    # Improved Local Search:
    # 1. Random Init
    # 2. Evaluate Score
    # 3. Identify the first gap x_gap.
    # 4. The gap x_gap has coverage c < 2. We need to increase coverage at x_gap.
    # 5. We can do this by changing the residue of some prime p_j to x_gap % p_j.
    # 6. Which prime to change? 
    #    - Changing a prime that currently covers x_gap is useless (coverage stays same or drops elsewhere).
    #    - We should pick a prime that does NOT cover x_gap.
    #    - Among those, pick one that minimizes the "damage" to other positions in the prefix.
    #    - "Damage" is the number of positions y in 1..x_gap-1 that currently have coverage == 2 
    #      and would lose coverage if we change p_j.
    
    def find_best_move(residues, current_L):
        """
        Finds a move that extends current_L or maintains it with less "fragility".
        Returns (new_residues, new_score)
        """
        if time.time() > deadline:
            return None, -1
            
        cover = compute_coverage_array(residues, current_L)
        
        # Find first gap
        gaps = np.where(cover[1:] < 2)[0]
        if len(gaps) == 0:
            # No gap in 1..current_L. Try to extend?
            # For simplicity in this step, if no gap, we try to find a move that keeps coverage >= 2
            # for a slightly larger range, or just return current.
            # Let's just try to extend the check range slightly if we have time.
            return None, current_L
            
        x_gap = gaps[0] + 1 # The actual integer x that is the first gap
        
        # Primes that do NOT cover x_gap
        candidates = []
        for i, p in enumerate(primes):
            req_r = x_gap % p
            if residues[i] != req_r:
                candidates.append(i)
        
        if not candidates:
            return None, current_L
            
        best_score = current_L - 1 # We want to improve
        best_res = None
        
        # We can't try all candidates if there are many (up to 168).
        # But 168 is small. We can evaluate the impact of each.
        # Impact of changing prime i to req_r:
        # 1. It adds coverage to x_gap (good).
        # 2. It removes coverage from old positions: old_r, old_r + p, ...
        # 3. It adds coverage to new positions: req_r, req_r + p, ...
        
        # To be fast, we only care about positions in 1..x_gap.
        # If we change prime i, coverage at x_gap increases by 1.
        # We need to ensure no position y in 1..x_gap drops below 2.
        # A position y drops below 2 if:
        #   - It currently has coverage 2.
        #   - It was covered by prime i with old_r (so y % p == old_r).
        #   - It is NOT covered by prime i with new_r (so y % p != req_r).
        #   - Note: if y % p == old_r AND y % p == req_r, then old_r == req_r, which is excluded.
        
        # So, for each candidate i:
        #   Find y in 1..x_gap such that:
        #     cover[y] == 2
        #     y % p_i == old_r_i
        #     y % p_i != req_r_i (always true if old != req)
        #   If such y exists, changing i breaks y (coverage becomes 1).
        #   So we must avoid candidates that break any "critical" point.
        
        # Critical points: y in 1..x_gap where cover[y] == 2.
        critical_mask = (cover[1:x_gap+1] == 2)
        critical_indices = np.where(critical_mask)[0] + 1
        
        if len(critical_indices) == 0:
            # No critical points. Any move that fixes x_gap is safe!
            # Just pick a random candidate.
            i = random.choice(candidates)
            new_res = residues[:]
            new_res[i] = x_gap % primes[i]
            new_score = get_score(new_res, max(current_L, x_gap + 10))
            return new_res, new_score
            
        # If there are critical points, we need to check which candidates are "safe".
        # A candidate i is safe if NO critical point y satisfies y % p_i == old_r_i.
        
        safe_candidates = []
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            
            # Check if any critical y is covered by (p, old_r)
            # y % p == old_r
            # We can check this by iterating critical_indices or using vectorized mod.
            # critical_indices is an array.
            if len(critical_indices) > 0:
                mods = critical_indices % p
                if np.any(mods == old_r):
                    # This move would break at least one critical point.
                    continue
            
            safe_candidates.append(i)
        
        if safe_candidates:
            # Pick one that maximizes "future potential"? 
            # For now, pick random safe candidate.
            i = random.choice(safe_candidates)
            new_res = residues[:]
            new_res[i] = x_gap % primes[i]
            new_score = get_score(new_res, max(current_L, x_gap + 10))
            return new_res, new_score
            
        # No safe candidate found.
        # We are stuck. We must break a critical point to fix x_gap.
        # This means we will lose some prefix length, but maybe gain more later?
        # Or we try a "repair" move: fix x_gap, then immediately fix the broken critical point?
        # This is a 2-step lookahead.
        
        # For now, let's try the candidate that breaks the FEWEST critical points.
        best_i = -1
        min_breaks = float('inf')
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            
            if len(critical_indices) > 0:
                mods = critical_indices % p
                breaks = np.sum(mods == old_r)
            else:
                breaks = 0
                
            if breaks < min_breaks:
                min_breaks = breaks
                best_i = i
                
        if best_i != -1 and min_breaks < 5: # Only accept if we break very few
            new_res = residues[:]
            new_res[best_i] = x_gap % primes[best_i]
            new_score = get_score(new_res, max(current_L, x_gap + 10))
            return new_res, new_score
            
        return None, current_L

    # Main Loop
    best_residues = [random.randint(0, p - 1) for p in primes]
    best_score = get_score(best_residues, 2000)
    
    # Try to improve
    # We will use a "Walk" strategy:
    # If we can't find a safe move, we might need to randomize a bit to escape local minima.
    
    while time.time() < deadline:
        new_res, new_score = find_best_move(best_residues, best_score)
        
        if new_res is not None:
            if new_score > best_score:
                best_residues = new_res
                best_score = new_score
                if best_score >= 1000:
                    break
            elif new_score == best_score:
                # Accept equal score moves with some probability to wander?
                # Or just take it if it changes the configuration (to allow future moves).
                if random.random() < 0.5:
                    best_residues = new_res
        else:
            # Stuck. Try a random perturbation.
            # Pick 1-2 random primes and change their residues.
            for _ in range(random.randint(1, 3)):
                i = random.randint(0, n_primes - 1)
                best_residues[i] = random.randint(0, primes[i] - 1)
            
            perturbed_score = get_score(best_residues, max(best_score, 2000))
            if perturbed_score > best_score:
                best_score = perturbed_score
            # If perturbed score is much worse, revert?
            # We don't have the old best_residues saved separately in this loop structure easily.
            # Let's keep a copy.
            
    # Final check
    # Ensure we return valid Python ints
    return [int(r) for r in best_residues]

# Quick self-test logic to ensure no crashes
if __name__ == "__main__":
    res = construct()
    print(f"Score: {len(res)} residues")
