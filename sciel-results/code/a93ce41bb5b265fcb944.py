import math
import random
import time
import sys

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

# Precompute primes and prime factors
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _get_primes(MAX_MOD)

def _prime_factors(n):
    factors = []
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            factors.append(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.append(n)
    return factors

# Precompute prime factors for all moduli
MOD_PRIMES = {m: _prime_factors(m) for m in range(2, MAX_MOD + 1)}

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _try_assign_residues(moduli, time_limit=None):
    """
    Attempt to assign residues to the given list of moduli such that
    all pairs are disjoint. Returns a list of (residue, modulus) if successful,
    or None if failed.
    Uses backtracking with forward checking.
    """
    if not moduli:
        return []
    
    # Sort moduli by "constraint density" to fail fast
    # Heuristic: moduli with more small prime factors are more constrained
    def score(m):
        return sum(1 / p for p in MOD_PRIMES[m])
    
    ordered_mods = sorted(moduli, key=score, reverse=True)
    
    assignment = {}
    
    def dfs(idx):
        if time_limit and time.time() > time_limit:
            return False
        if idx == len(ordered_mods):
            return True
        
        m = ordered_mods[idx]
        # Try all residues
        # Optimization: We can restrict residues based on previous assignments
        # For each previous modulus m_prev, we have a constraint on r mod gcd(m, m_prev)
        
        # Simple approach: try all 0..m-1
        # To speed up, we can check against already assigned moduli
        for r in range(m):
            ok = True
            for m_prev in ordered_mods[:idx]:
                r_prev = assignment[m_prev]
                if not _disjoint(r, m, r_prev, m_prev):
                    ok = False
                    break
            if ok:
                assignment[m] = r
                if dfs(idx + 1):
                    return True
                del assignment[m]
        
        return False

    if dfs(0):
        return [(assignment[m], m) for m in moduli]
    return None

def _construct_with_prime_bounds(order, time_limit, rng):
    """
    Greedy construction respecting prime count bounds.
    """
    selected_mods = []
    prime_counts = {p: 0 for p in PRIMES}
    max_counts = {p: p for p in PRIMES} # Max p multiples of p
    
    # We will store the final pairs here
    pairs = []
    
    # We need to be able to check validity of the whole set at the end,
    # but during greedy we assume we can assign residues.
    # To be safe, we will periodically verify with _try_assign_residues.
    
    for m in order:
        if time.time() > time_limit:
            break
            
        # Check prime count bounds
        can_add = True
        for p in MOD_PRIMES[m]:
            if prime_counts[p] >= max_counts[p]:
                can_add = False
                break
        
        if not can_add:
            continue
            
        # Try to add m
        # Tentatively update counts
        for p in MOD_PRIMES[m]:
            prime_counts[p] += 1
            
        # Check if we can assign residues to the new set
        # To save time, we only do full backtracking every K steps or if set is small
        # For now, let's just track the moduli. We will verify at the end of the pass.
        selected_mods.append(m)
        
    # After greedy pass, verify if residues can be assigned
    # If not, we need to remove some moduli.
    # This is a "repair" step.
    
    # Simple repair: Remove moduli until it works?
    # Better: Use local search on the set of moduli.
    
    return selected_mods

def _local_search_moduli(moduli_set, all_mods, time_limit, rng):
    """
    Local search on the set of moduli.
    Moves: Swap one modulus for another.
    Objective: Maximize size, then ensure residue assignability.
    """
    current_mods = list(moduli_set)
    current_pairs = _try_assign_residues(current_mods, time_limit)
    
    if current_pairs is None:
        # If initial set is invalid, try to fix by removing
        # This shouldn't happen if we start from a valid greedy solution
        return []

    best_pairs = current_pairs
    best_mods = current_mods
    best_size = len(best_mods)
    
    # Precompute prime counts for best_mods
    def get_prime_counts(mods):
        counts = {p: 0 for p in PRIMES}
        for m in mods:
            for p in MOD_PRIMES[m]:
                counts[p] += 1
        return counts

    current_counts = get_prime_counts(best_mods)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        used_mods = set(best_mods)
        missed = [m for m in all_mods if m not in used_mods]
        
        if not missed:
            break
            
        # Shuffle missed for randomness
        rng.shuffle(missed)
        
        for m_new in missed:
            if time.time() > time_limit:
                break
                
            # Check if m_new can be added by swapping with some m_old
            # Identify bottleneck primes for m_new
            # If m_new has a prime p where count is full, we must remove a multiple of p.
            
            candidate_removals = []
            for p in MOD_PRIMES[m_new]:
                if current_counts[p] >= p: # Full
                    # Find all selected moduli that are multiples of p
                    for m_old in best_mods:
                        if p in MOD_PRIMES[m_old]:
                            candidate_removals.append(m_old)
            
            # If no bottleneck, we can just add?
            # But we have distinct moduli constraint. If we add, size increases.
            # If we can add without removing, that's a win.
            # Check if we can add m_new directly
            # This requires checking prime bounds.
            can_add_directly = True
            for p in MOD_PRIMES[m_new]:
                if current_counts[p] >= p:
                    can_add_directly = False
                    break
            
            if can_add_directly:
                # Try to add
                trial_mods = best_mods + [m_new]
                trial_pairs = _try_assign_residues(trial_mods, time_limit)
                if trial_pairs is not None:
                    best_mods = trial_mods
                    best_pairs = trial_pairs
                    best_size += 1
                    current_counts = get_prime_counts(best_mods)
                    improved = True
                    break # Restart loop with new best
            
            # If direct add failed or not possible, try swaps
            if not candidate_removals:
                # No specific bottleneck, try random swap?
                # This is less efficient. Skip for now.
                continue
                
            rng.shuffle(candidate_removals)
            
            for m_old in candidate_removals:
                if time.time() > time_limit:
                    break
                    
                # Perform swap
                trial_mods = [m for m in best_mods if m != m_old] + [m_new]
                # Check prime bounds for trial_mods?
                # The swap respects the bottleneck p, but might violate others?
                # m_old and m_new might have different prime factors.
                # We should check bounds.
                valid_bounds = True
                trial_counts = get_prime_counts(trial_mods)
                for p in PRIMES:
                    if trial_counts[p] > p:
                        valid_bounds = False
                        break
                
                if not valid_bounds:
                    continue
                    
                trial_pairs = _try_assign_residues(trial_mods, time_limit)
                if trial_pairs is not None:
                    # Accept swap
                    best_mods = trial_mods
                    best_pairs = trial_pairs
                    # Size remains same
                    current_counts = trial_counts
                    improved = True
                    break # Restart loop
            
            if improved:
                break
                
    return best_pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Precompute heuristic ordering
    # Sort by number of prime factors (desc) or sum of 1/p (desc)
    def constraint_score(m):
        return sum(1.0 / p for p in MOD_PRIMES[m])
    
    order_most_constrained = sorted(all_mods, key=constraint_score, reverse=True)
    
    best_pairs = []
    best_len = 0
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts)
        
        # Create a slightly perturbed order
        order = order_most_constrained[:]
        # Randomly swap a few elements
        for _ in range(5):
            i, j = rng.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
            
        # Greedy phase
        local_search_time = deadline - 2.0
        greedy_mods = _construct_with_prime_bounds(order, local_search_time - 5.0, rng)
        
        # If greedy failed to assign residues, try to fix
        # _construct_with_prime_bounds returns moduli, not pairs.
        # We need to ensure they are valid.
        # Let's verify and repair if necessary.
        
        # For now, just try local search on the greedy result.
        # If greedy result is empty or invalid, local search might fail.
        # Let's ensure we have a valid starting point.
        
        # Simple check:
        start_pairs = _try_assign_residues(greedy_mods, local_search_time - 5.0)
        if start_pairs is None:
            # Fallback: try a smaller set or random valid set
            # Just pick a few primes
            start_pairs = [(0, p) for p in PRIMES if p <= MAX_MOD]
            # Verify
            if not all(_disjoint(r1, m1, r2, m2) for i, (r1, m1) in enumerate(start_pairs) for (r2, m2) in start_pairs[i+1:]):
                # If even primes fail (shouldn't), return empty
                return []
        
        # Local Search
        pairs = _local_search_moduli([m for _, m in start_pairs], all_mods, local_search_time, rng)
        
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_pairs = pairs
            
        if best_len == len(all_mods):
            break
            
    return [[r, m] for r, m in best_pairs]
