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
    time_limit = 118.0  # Safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    MAX_CHECK = 10000
    
    def build_counts(residues):
        """Build coverage count array from residues."""
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                idx = np.arange(start, MAX_CHECK + 1, p)
                counts[idx] += 1
        return counts

    def get_first_fail(counts):
        """Return first index >= 1 where count < 2, or MAX_CHECK if none."""
        fail_mask = counts < 2
        if np.any(fail_mask[1:]):
            first_fail = int(np.argmax(fail_mask[1:])) + 1
            return first_fail
        else:
            return MAX_CHECK

    def score_from_counts(counts):
        """Compute score (largest L with all 1..L covered >= 2 times)."""
        first_fail = get_first_fail(counts)
        return first_fail - 1

    def apply_move(counts, residues, prime_idx, new_r):
        """Apply a move: change residues[prime_idx] to new_r, update counts in place."""
        p = primes[prime_idx]
        old_r = residues[prime_idx]
        if old_r == new_r:
            return
        # Remove old residue's contributions
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= MAX_CHECK:
            idx_old = np.arange(start_old, MAX_CHECK + 1, p)
            counts[idx_old] -= 1
        # Add new residue's contributions
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= MAX_CHECK:
            idx_new = np.arange(start_new, MAX_CHECK + 1, p)
            counts[idx_new] += 1
        residues[prime_idx] = new_r

    def is_valid_move(counts, prime_idx, new_r, x_fail):
        """Check if changing prime_idx to new_r keeps prefix [1, x_fail-1] valid."""
        p = primes[prime_idx]
        old_r = counts._old_r if hasattr(counts, '_old_r') else None
        # We need to pass old_r separately
        return _check_valid(counts, p, old_r, new_r, x_fail)

    def _check_valid(counts, p, old_r, new_r, x_fail):
        """Check validity of a move given old and new residues."""
        if old_r == new_r:
            return True
        # Only y where y % p == old_r and y % p != new_r can drop in count
        # We need counts[y] >= 2 for all such y in [1, x_fail-1]
        if old_r == 0:
            start_y = p
        else:
            start_y = old_r
        if start_y >= x_fail:
            return True
        # Check all y in [start_y, x_fail) with step p
        for y in range(start_y, x_fail, p):
            if counts[y] <= 1:
                return False
        return True

    def is_move_valid(counts, residues, prime_idx, new_r, x_fail):
        """Check if changing residues[prime_idx] to new_r keeps prefix [1, x_fail-1] valid."""
        p = primes[prime_idx]
        old_r = residues[prime_idx]
        if old_r == new_r:
            return True
        if old_r == 0:
            start_y = p
        else:
            start_y = old_r
        if start_y >= x_fail:
            return True
        for y in range(start_y, x_fail, p):
            if counts[y] <= 1:
                return False
        return True

    best_score = 0
    best_residues = None
    
    # Strategy: multiple restarts with incremental local search
    # Use incremental count updates for speed
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 3:
            break
            
        # Initialize residues with some structure: try to cover small numbers well
        # Use a heuristic: for small primes, pick residues that cover 1,2,3,...
        residues = [0] * n_primes
        for i, p in enumerate(primes):
            # For each prime, pick a residue that helps cover the current gap
            # Simple heuristic: r = i % p or random
            residues[i] = int(np.random.randint(0, p))
        
        # Build initial counts
        counts = build_counts(residues)
        current_score = score_from_counts(counts)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
        
        # Local search: iteratively fix the first failure
        max_no_improve = 0
        no_improve_limit = 40
        
        while time.time() - start_time < time_limit - 3:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure
            x_fail = get_first_fail(counts)
            if x_fail >= MAX_CHECK:
                # Fully covered up to MAX_CHECK
                current_score = MAX_CHECK - 1
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                break
                
            # Try to fix x_fail by changing one prime's residue
            move_made = False
            improved = False
            
            # Try primes in order of increasing size (smaller primes affect more numbers)
            # But also consider that we want to cover x_fail specifically
            # new_r for prime p should be x_fail % p to cover x_fail
            
            # Try all primes, but prioritize those where the move is "cheap"
            # (i.e., the old residue doesn't cover many critical points)
            prime_order = list(range(n_primes))
            # Sort by prime value (smaller first)
            prime_order.sort(key=lambda i: primes[i])
            
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                if is_move_valid(counts, residues, i, new_r, x_fail):
                    # Apply move incrementally
                    old_r = residues[i]
                    apply_move(counts, residues, i, new_r)
                    
                    # Recompute score from updated counts
                    new_score = score_from_counts(counts)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        move_made = True
                        improved = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score:
                        # Accept neutral moves to escape local minima
                        if max_no_improve > 3:
                            move_made = True
                            max_no_improve += 1
                            break
                        else:
                            # Revert
                            apply_move(counts, residues, i, old_r)
                    else:
                        # Revert
                        apply_move(counts, residues, i, old_r)
                        
            if not move_made:
                # Stuck. Try random perturbation
                escaped = False
                for _ in range(30):
                    if time.time() - start_time > time_limit - 3:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    # Apply and check
                    apply_move(counts, residues, i, new_r)
                    new_score = score_from_counts(counts)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 8:
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        apply_move(counts, residues, i, old_r)
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break
        
        # End of local search for this restart

    # Final polish: try to extend the best solution further with more aggressive search
    if best_residues is not None:
        residues = best_residues[:]
        counts = build_counts(residues)
        current_score = score_from_counts(counts)
        if current_score < best_score:
            current_score = best_score
            
        # Try to improve further with more careful search
        polish_iters = 0
        while time.time() - start_time < time_limit - 1:
            polish_iters += 1
            x_fail = get_first_fail(counts)
            if x_fail >= MAX_CHECK:
                current_score = MAX_CHECK - 1
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                break
                
            # Try all primes
            prime_order = list(range(n_primes))
            prime_order.sort(key=lambda i: primes[i])
            
            moved = False
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                if is_move_valid(counts, residues, i, new_r, x_fail):
                    old_r = residues[i]
                    apply_move(counts, residues, i, new_r)
                    new_score = score_from_counts(counts)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        moved = True
                        break
                    else:
                        apply_move(counts, residues, i, old_r)
                        
            if not moved:
                # Try some random moves to escape
                any_escape = False
                for _ in range(20):
                    if time.time() - start_time > time_limit - 1:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    apply_move(counts, residues, i, new_r)
                    new_score = score_from_counts(counts)
                    if new_score >= current_score:
                        if new_score > current_score:
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_residues = residues[:]
                        any_escape = True
                        break
                    else:
                        apply_move(counts, residues, i, old_r)
                if not any_escape:
                    break

    # Return the best residues found
    if best_residues is None:
        # Fallback
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
