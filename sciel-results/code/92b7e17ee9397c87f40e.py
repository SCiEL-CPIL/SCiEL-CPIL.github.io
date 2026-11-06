import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    primes = _get_primes(1000)
    n_primes = len(primes)

    # We will search for L up to a reasonable max.
    MAX_CHECK = 20000

    # Precompute indices for each prime and residue
    indices_cache = [None] * n_primes
    for i, p in enumerate(primes):
        indices_cache[i] = [None] * p
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                indices_cache[i][r] = np.arange(start, MAX_CHECK + 1, p, dtype=np.int32)
            else:
                indices_cache[i][r] = np.array([], dtype=np.int32)

    def calculate_score(residues):
        """
        Calculate the largest L such that every integer 1..L satisfies at least two congruences.
        """
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = indices_cache[i][r]
            if len(idx) > 0:
                counts[idx] += 1

        # Find the first index where count < 2
        # We care about 1..L
        if MAX_CHECK == 0:
            return 0

        # Create a boolean array where counts < 2
        fail_mask = counts < 2
        # We only care about indices 1..MAX_CHECK
        # If fail_mask[1:] has any True, find the first one.
        if np.any(fail_mask[1:]):
            first_fail = np.argmax(fail_mask[1:]) + 1
            return first_fail - 1
        else:
            return MAX_CHECK

    def sat_search(target_L, initial_residues=None, time_budget=5.0):
        """
        DPLL-style backtracking with unit propagation to find residues such that
        every integer 1..target_L satisfies at least two congruences.
        
        Returns (residues, success) where success is True if found.
        """
        deadline = time.time() + time_budget
        
        # Variables: residues[i] for i in range(n_primes)
        # Domain: 0..p-1 for prime i
        
        # Precompute for each x in 1..target_L, which (prime_idx, residue) pairs cover it
        # cover_map[x] = list of (i, r) such that x % p_i == r
        # But we can be smarter: for each x, we need at least 2 covers.
        # A cover is determined by choosing residue[i] = x % p_i.
        
        # Let's use a different approach: maintain counts of how many primes cover each x.
        # But in SAT, we don't have counts, we have assignments.
        
        # Simpler: backtracking with constraint propagation.
        # State: assignment (list of residues or None)
        # Constraint: for each x in 1..target_L, at least 2 primes i have assignment[i] == x % p_i.
        
        # We'll use a recursive DPLL with:
        # 1. Unit propagation: if some x has only 1 possible cover left, force it.
        # 2. If some x has 0 possible covers, fail.
        # 3. Choose a prime with smallest domain, try values.
        
        # To make it tractable, we'll use a heuristic: process primes in order of increasing p
        # (smaller primes cover more numbers, so they're more critical).
        
        # Actually, let's use a different representation:
        # For each x, track which primes could still cover it.
        # A prime i can cover x if assignment[i] is None or assignment[i] == x % p_i.
        
        # This is complex. Let's use a simpler approach:
        # Greedy with backtracking on the first failure.
        
        if initial_residues is None:
            assignment = [None] * n_primes
        else:
            assignment = initial_residues[:]
        
        # Precompute for each prime i, the residue that covers each x
        # cover_residue[i][x] = x % p_i (if x is in range)
        # But we don't need to precompute; just compute on the fly.
        
        # We'll use a stack-based DFS to avoid recursion depth issues.
        # State: (assignment, next_prime_idx)
        # But we need to track which x are satisfied.
        
        # Alternative: use the local search idea but with SAT-style propagation.
        # Let's try a different approach: incremental constraint satisfaction.
        
        # Start with all residues unassigned.
        # For each x from 1 to target_L, we need at least 2 primes to cover it.
        # This is a covering problem.
        
        # Let's use a greedy algorithm with backtracking:
        # 1. Try to assign residues to primes one by one.
        # 2. After each assignment, check if any x in 1..target_L has < 2 covers.
        # 3. If so, backtrack.
        
        # To make it efficient, we'll only check x that are affected by the new assignment.
        
        # Precompute: for each prime i and residue r, which x in 1..target_L are covered?
        # covered_by[i][r] = set of x in 1..target_L such that x % p_i == r
        
        # But this is memory intensive. Let's compute on the fly.
        
        # Actually, let's use the existing indices_cache but limit to target_L.
        
        # For now, let's implement a simple backtracking that tries to extend the prefix.
        
        # We'll maintain:
        # - assignment: list of residues or None
        # - counts: array of length target_L+1, counts[x] = number of primes covering x
        # - possible_covers[x]: set of primes that could still cover x (i.e., assignment[i] is None or assignment[i] == x % p_i)
        
        # This is getting complex. Let's simplify:
        
        # Use a different strategy: local search with SAT-style constraint checking.
        # Start with a random or greedy assignment.
        # Find the first x where count[x] < 2.
        # Try to fix it by changing some prime's residue.
        # Use constraint propagation to avoid breaking other x's.
        
        # This is similar to the existing local search but with better move selection.
        
        # For now, let's implement a simple version and see if it helps.
        
        # Actually, let's try a completely different approach:
        # Use integer programming or a different optimization method.
        
        # But we only have numpy and scipy. Let's use scipy.optimize.
        
        # Define the problem as: maximize L such that for all x in 1..L, sum_i [residues[i] == x % p_i] >= 2.
        # This is a non-convex optimization problem.
        
        # Let's use a simulated annealing approach.
        
        # For now, let's just return the initial residues if no improvement is found.
        
        return assignment, False
    
    # For now, let's keep the existing local search but improve it.
    # The SAT approach is too complex to implement quickly.
    
    def local_search(residues, current_score, time_budget):
        max_no_improve = 0
        no_improve_limit = 50
        
        while time.time() - start_time < time_budget:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure
            counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                idx = indices_cache[i][r]
                if len(idx) > 0:
                    counts[idx] += 1
                    
            fail_mask = counts < 2
            if not np.any(fail_mask[1:]):
                if current_score < MAX_CHECK:
                    current_score = MAX_CHECK
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try to fix x_fail - try multiple candidate moves
            move_made = False
            improved = False
            
            # Try each prime, preferring smaller primes first (they affect more numbers)
            prime_order = sorted(range(n_primes), key=lambda i: primes[i])
            
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                old_indices = indices_cache[i][residues[i]]
                if len(old_indices) > 0:
                    cutoff_idx = np.searchsorted(old_indices, x_fail, side='left')
                    if cutoff_idx > 0:
                        affected_counts = counts[old_indices[:cutoff_idx]]
                        if np.any(affected_counts == 2):
                            continue
                
                # Apply move
                old_r = residues[i]
                residues[i] = new_r
                # Recalculate score
                new_score = calculate_score(residues)
                
                if new_score > current_score:
                    current_score = new_score
                    move_made = True
                    improved = True
                    max_no_improve = 0
                    break # Restart the while loop to find new first failure
                elif new_score == current_score:
                    # Accept neutral moves to escape local minima
                    if max_no_improve > 5:
                        move_made = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                else:
                    # Revert
                    residues[i] = old_r
                    
            if not move_made:
                # Stuck. Try random perturbation
                escaped = False
                for _ in range(20):
                    if time.time() - start_time > time_budget:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    new_score = calculate_score(residues)
                    if new_score > current_score:
                        current_score = new_score
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 10:
                        # Accept neutral move
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break # Restart this search instance
        
        return residues, current_score

    # Start with all zeros
    residues = [0] * n_primes
    best_score = calculate_score(residues)
    best_residues = residues[:]
    
    # Run local search
    residues, score = local_search(residues[:], best_score, time_limit - 10)
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    # Try a few random restarts
    if time.time() - start_time < time_limit - 20:
        for restart in range(3):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Random initialization
            residues = [int(np.random.randint(0, p)) for p in primes]
            current_score = calculate_score(residues)
            
            # Local search
            residues, score = local_search(residues, current_score, time_limit - 5)
            
            if score > best_score:
                best_score = score
                best_residues = residues[:]
    
    # Return the best residues found
    if best_residues is None:
        # Fallback
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
