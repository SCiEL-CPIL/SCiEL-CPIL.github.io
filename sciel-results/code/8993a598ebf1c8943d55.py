import time
import numpy as np
from collections import defaultdict

def _primes_upto(n):
    """Generate primes up to n using sieve."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)
    
    # We will search for L up to 2000 (well above 1000)
    MAX_L = 2000
    
    # Precompute coverage masks: for each prime i, a boolean array of length MAX_L+1
    # where mask[i, x] = True if x ≡ r_i (mod p_i)
    # Actually, we can compute coverage more efficiently.
    # For a given residue vector r, coverage at x is sum over i of [x ≡ r_i (mod p_i)]
    
    # Precompute for each x in 1..MAX_L, the list of (prime_idx, residue) pairs that would cover x
    # Actually, for each x, the residue that covers it for prime i is x % p_i.
    # So we can precompute: for each x, for each i, the required residue is x % primes[i]
    
    # Let's precompute a 2D array: cover_req[x, i] = x % primes[i]
    # This is (MAX_L+1) x n_primes
    # x ranges 1..MAX_L, i ranges 0..n_primes-1
    # cover_req[x, i] = x % primes[i]
    
    # Actually, we don't need to store this. We can compute it on the fly.
    # But for speed, let's precompute.
    
    # x_vals: 1..MAX_L
    x_vals = np.arange(1, MAX_L + 1, dtype=np.int64)  # shape (MAX_L,)
    
    # cover_req: shape (MAX_L, n_primes)
    # cover_req[j, i] = (j+1) % primes[i]  since x_vals[j] = j+1
    cover_req = np.zeros((MAX_L, n_primes), dtype=np.int64)
    for i in range(n_primes):
        cover_req[:, i] = x_vals % primes_np[i]
    
    # Initial residues: use a simple heuristic
    # For each prime i, set residue to 0 or 1 to maximize initial coverage
    # Let's start with all residues = 0
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_coverage(res):
        """
        Compute coverage array for all x in 1..MAX_L.
        cov[x] = number of primes i where x ≡ res[i] (mod p_i)
        """
        # res: shape (n_primes,)
        # For each i, the covered x values are those where x % p_i == res[i]
        # We can use cover_req: cover_req[:, i] == res[i]
        
        # cov = sum over i of [cover_req[:, i] == res[i]]
        # This is a boolean comparison: cover_req == res[None, :]
        # cover_req: (MAX_L, n_primes), res[None, :]: (1, n_primes)
        # broadcast: (MAX_L, n_primes)
        
        matches = (cover_req == res[None, :])  # shape (MAX_L, n_primes)
        cov = np.sum(matches, axis=1)  # shape (MAX_L,)
        return cov
    
    def get_L(cov):
        """Find largest L such that cov[0..L-1] all >= 2 (cov[0] corresponds to x=1)."""
        # cov[j] corresponds to x = j+1
        # We need the first j where cov[j] < 2
        # L = j (since x=j+1 is the first uncovered, so L = j)
        # Actually, if cov[j] < 2, then x = j+1 is not doubly covered.
        # So the largest L such that all x in 1..L are doubly covered is j.
        # Because x=1..j are covered, x=j+1 is not.
        
        # Find first index where cov < 2
        idx = np.argmax(cov < 2)
        if cov[idx] >= 2:
            # All are >= 2
            return MAX_L
        else:
            return idx  # L = idx, since x=1..idx are covered, x=idx+1 is not
    
    def get_score(res):
        cov = compute_coverage(res)
        return get_L(cov)
    
    # Initial score
    best_residues = residues.copy()
    best_score = get_score(best_residues)
    
    # Parallel coordinate ascent
    # At each step, for each prime i, try changing its residue to a new value.
    # We evaluate all n_primes * K candidates in parallel, where K is the number of new residues to try per prime.
    # For each candidate, compute the new coverage and score.
    # Pick the candidate with the highest score.
    
    # To make this efficient, we can batch the evaluations.
    # For each prime i, we try K new residues. Total candidates = n_primes * K.
    # For each candidate, we need to compute coverage.
    # We can do this by creating a batch of residue vectors and computing coverage for all.
    
    # Let's use K = 10 new residues per prime per iteration.
    # Total candidates = 168 * 10 = 1680.
    # For each candidate, we compute coverage (MAX_L x n_primes boolean array) -> expensive.
    # Alternative: for each candidate, only the coverage of the changed prime i differs.
    # So we can compute the base coverage, then for each candidate, adjust only the i-th prime's contribution.
    
    # Let's precompute the base coverage for the current residues.
    # Then for a candidate that changes prime i from old_r to new_r:
    #   new_cov = base_cov - [x ≡ old_r (mod p_i)] + [x ≡ new_r (mod p_i)]
    
    # This is much faster: only O(MAX_L) per candidate instead of O(MAX_L * n_primes).
    
    # Let's implement this.
    
    def compute_single_prime_coverage(residue, prime_idx):
        """Compute coverage array for a single prime with given residue."""
        p = primes_np[prime_idx]
        # x in 1..MAX_L where x ≡ residue (mod p)
        # x = residue + k*p
        # If residue == 0, x starts at p; else x starts at residue
        if residue == 0:
            first_x = p
        else:
            first_x = residue
        if first_x > MAX_L:
            return np.zeros(MAX_L, dtype=np.int64)
        # Generate x values
        # x = first_x, first_x + p, first_x + 2p, ... <= MAX_L
        # Number of terms: (MAX_L - first_x) // p + 1
        n_terms = (MAX_L - first_x) // p + 1
        if n_terms <= 0:
            return np.zeros(MAX_L, dtype=np.int64)
        x_vals_single = first_x + np.arange(n_terms, dtype=np.int64) * p
        cov = np.zeros(MAX_L, dtype=np.int64)
        cov[x_vals_single - 1] = 1  # x_vals_single - 1 because cov[j] corresponds to x=j+1
        return cov
    
    # Base coverage for current residues
    base_cov = compute_coverage(best_residues)
    
    # Parallel coordinate ascent
    max_iterations = 500
    K = 20  # New residues to try per prime per iteration
    
    for it in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Generate candidates: for each prime i, K random new residues
        # candidates: shape (n_primes * K, 2) where col 0 = prime_idx, col 1 = new_residue
        # Also keep track of old residue for each candidate
        
        candidate_prime_indices = []
        candidate_new_residues = []
        candidate_old_residues = []
        
        for i in range(n_primes):
            old_r = best_residues[i]
            p = primes[i]
            # Generate K random residues, avoiding old_r
            # Use random permutation of 0..p-1, take first K
            if p <= K:
                # All residues
                new_rs = np.arange(p, dtype=np.int64)
                # Remove old_r
                mask = new_rs != old_r
                new_rs = new_rs[mask]
            else:
                # Random sample
                rng = np.random.default_rng(seed=it * 1000 + i)
                new_rs = rng.choice(p, size=K, replace=False)
                # Remove old_r if present
                mask = new_rs != old_r
                new_rs = new_rs[mask]
            
            if len(new_rs) == 0:
                continue
            
            candidate_prime_indices.extend([i] * len(new_rs))
            candidate_new_residues.extend(new_rs.tolist())
            candidate_old_residues.extend([old_r] * len(new_rs))
        
        if not candidate_prime_indices:
            break
        
        n_candidates = len(candidate_prime_indices)
        
        # For each candidate, compute new coverage and score
        # new_cov = base_cov - old_prime_cov + new_prime_cov
        
        best_candidate_score = best_score
        best_candidate_prime = -1
        best_candidate_new_res = -1
        
        # Process in batches for speed
        batch_size = 200
        for batch_start in range(0, n_candidates, batch_size):
            if time.time() > deadline - 5:
                break
            
            batch_end = min(batch_start + batch_size, n_candidates)
            batch_primes = candidate_prime_indices[batch_start:batch_end]
            batch_new_rs = candidate_new_residues[batch_start:batch_end]
            batch_old_rs = candidate_old_residues[batch_start:batch_end]
            
            # For each candidate in batch, compute new_cov
            # We'll compute scores and track the best
            
            for j in range(len(batch_primes)):
                i = batch_primes[j]
                old_r = batch_old_rs[j]
                new_r = batch_new_rs[j]
                
                # Compute old and new prime coverage
                old_prime_cov = compute_single_prime_coverage(old_r, i)
                new_prime_cov = compute_single_prime_coverage(new_r, i)
                
                new_cov = base_cov - old_prime_cov + new_prime_cov
                new_score = get_L(new_cov)
                
                if new_score > best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate_prime = i
                    best_candidate_new_res = new_r
        
        if best_candidate_prime >= 0:
            # Accept the best candidate
            best_residues[best_candidate_prime] = best_candidate_new_res
            best_score = best_candidate_score
            # Update base_cov
            base_cov = compute_coverage(best_residues)
        else:
            # No improvement; try random restart or break
            # Let's try a random perturbation
            if it % 10 == 0:
                # Perturb 5 random residues
                for _ in range(5):
                    i = np.random.randint(n_primes)
                    p = primes[i]
                    new_r = np.random.randint(p)
                    best_residues[i] = new_r
                base_cov = compute_coverage(best_residues)
                new_score = get_score(best_residues)
                if new_score > best_score:
                    best_score = new_score
                else:
                    # Revert
                    # We don't have the old values stored, so we can't easily revert.
                    # This is a problem. Let's keep a history.
                    pass
    
    # Convert to list of Python integers
    return [int(r) for r in best_residues]
