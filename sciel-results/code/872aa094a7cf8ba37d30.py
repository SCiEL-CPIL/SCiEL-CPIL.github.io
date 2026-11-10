import time
import math
import numpy as np
from itertools import product

N = 5001  # Range [0, 5000]
DEADLINE = 290.0  # Safety margin before 330s limit

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# We need primes up to N to guarantee validity for any subset of [0, N-1]
# The condition is: for every prime p <= len(S), some residue is missed.
# If we handle all primes p <= N, we satisfy the condition for any possible S.
PRIMES = _primes(N)
P_COUNT = len(PRIMES)

def _compute_score_and_cover(residues):
    """
    Computes cover_count and score for a given residue assignment.
    cover_count[i] is the number of primes p such that i % p == residues[p_idx].
    Score is the number of i with cover_count[i] == 0.
    """
    cover_count = np.zeros(N, dtype=np.int32)
    for i, p in enumerate(PRIMES):
        r = residues[i]
        if r < p:
            cover_count[r::p] += 1
    score = np.sum(cover_count == 0)
    return cover_count, score

def _greedy_fill(residues, start_idx, time_limit):
    """
    Greedily fills residues from start_idx to end.
    Returns updated residues list.
    """
    # We need an 'available' mask to know which elements are currently not covered
    # by the already chosen residues.
    # However, for greedy fill, we usually want to pick the residue that covers the FEWEST
    # currently available elements? No, that's for minimization.
    # Here, we want to MAXIMIZE the number of elements with cover_count 0.
    # So we want to MINIMIZE the number of elements that get covered by the new prime.
    # An element is 'available' if it is not covered by any previous prime.
    # Wait, the greedy logic in the inspiration was:
    # "pick residue for each prime that removes fewest available numbers"
    # This minimizes the reduction in the set size.
    
    # Let's implement this using a boolean mask of 'uncovered' elements.
    uncovered = np.ones(N, dtype=bool)
    
    # Apply previous residues to update 'uncovered'
    for i in range(start_idx):
        p = PRIMES[i]
        r = residues[i]
        if r < p:
            uncovered[r::p] = False
            
    for i in range(start_idx, P_COUNT):
        if time.time() > time_limit:
            break
        p = PRIMES[i]
        min_count = N
        best_r = 0
        # Find residue r that covers the fewest 'uncovered' elements
        # cover_count_in_uncovered[r] = sum(uncovered[r::p])
        # We want to minimize this sum.
        
        # Optimization: For large p, the sum is roughly N/p.
        # For small p, we can compute exactly.
        # Since N=5001, and we only do this once per branch (or restart), 
        # and p is small for the first few, it's fast.
        
        # Vectorized computation for all r is hard without loop because of strides.
        # But p is small for early primes.
        # For p > 100, the variation is small, so we can just pick 0 or random.
        # Let's loop for p <= 100, else pick 0.
        
        if p <= 100:
            for r in range(p):
                count = np.sum(uncovered[r::p])
                if count < min_count:
                    min_count = count
                    best_r = r
        else:
            best_r = 0 # Default choice for large p, greedy improvement will fix later
        
        residues[i] = best_r
        # Update uncovered mask
        uncovered[best_r::p] = False
        
    return residues

def _hill_climb(residues, cover_count, current_score, time_limit):
    """
    Hill Climbing: find best single move (swap residue for a prime), apply, repeat.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_move = None # (prime_idx, new_residue)
        
        for i, p in enumerate(PRIMES):
            old_r = residues[i]
            
            # Calculate gain from removing old_r coverage
            # Elements with cover_count == 1 in old_r::p will become 0 (good)
            # Elements with cover_count > 1 will decrease but stay > 0 (neutral)
            # Elements with cover_count == 0 are not in old_r::p (impossible if old_r covers them)
            
            # We only care about elements that are currently covered by old_r.
            # Indices: old_r, old_r+p, old_r+2p...
            old_indices = np.arange(old_r, N, p)
            
            # If no indices, skip
            if len(old_indices) == 0:
                continue
                
            # Gain: number of elements in old_indices with cover_count == 1
            gain = np.sum(cover_count[old_indices] == 1)
            
            if gain == 0:
                continue
                
            # Calculate loss for all other residues
            # We want to find r_new such that the number of elements in r_new::p
            # with cover_count == 0 is minimized.
            # These elements would become covered (cover_count becomes 1), reducing score.
            
            min_loss = N
            best_new_r = -1
            
            # Optimization:
            # If p is large, we can't loop over all r.
            # But for large p, the 'gain' is likely small, and 'loss' is likely small.
            # Let's loop if p <= 200.
            
            if p <= 200:
                for r in range(p):
                    if r == old_r:
                        continue
                    new_indices = np.arange(r, N, p)
                    if len(new_indices) == 0:
                        loss = 0
                    else:
                        loss = np.sum(cover_count[new_indices] == 0)
                    
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r
            else:
                # For large p, sample residues
                # The 'loss' is the number of zeros in the slice.
                # Zeros are sparse (score ~ 1000-2000 out of 5000).
                # We can just check a few candidates, e.g., r=0, r=1, r=2...
                # Or better, check residues that are likely to have few zeros.
                # Actually, for large p, the slice is short.
                # Let's just check r=0.
                # A smarter approach: check residues that align with existing zeros?
                # Too complex. Let's stick to r=0 for large p in hill climb,
                # relying on the initial greedy or branching to set it better.
                # Or, check a random sample.
                # Let's check r=0 and r=1.
                for r in [0, 1]:
                    if r == old_r:
                        continue
                    new_indices = np.arange(r, N, p)
                    if len(new_indices) == 0:
                        loss = 0
                    else:
                        loss = np.sum(cover_count[new_indices] == 0)
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r

            if best_new_r != -1:
                delta = gain - min_loss
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, best_new_r)
                    
        if best_move is not None:
            i, new_r = best_move
            p = PRIMES[i]
            old_r = residues[i]
            
            # Apply move
            residues[i] = new_r
            
            # Update cover_count
            # Remove old coverage
            cover_count[old_r::p] -= 1
            # Add new coverage
            cover_count[new_r::p] += 1
            
            current_score += best_delta
            improved = True
            
    return residues, cover_count, current_score

def construct():
    start_time = time.time()
    
    # Phase 1: Branch over first 5 primes to find a good starting point
    # Primes: 2, 3, 5, 7, 11
    branch_primes = PRIMES[:5]
    branch_ranges = [range(p) for p in branch_primes]
    
    best_score = -1
    best_residues = None
    best_cover = None
    
    # Budget for branching: 30 seconds
    branch_deadline = start_time + 30.0
    
    for combo in product(*branch_ranges):
        if time.time() > branch_deadline:
            break
            
        # Initialize residues
        residues = list(combo) + [0] * (P_COUNT - 5)
        
        # Greedy fill for remaining primes
        # We use a simplified greedy fill that doesn't check time per prime,
        # but the outer loop checks time.
        # This is fast enough for 5000 elements.
        
        # To speed up, we can pre-compute the 'uncovered' mask for the first 5 primes?
        # No, it depends on the combo.
        
        # Let's run the greedy fill
        residues = _greedy_fill(residues, 5, branch_deadline)
        
        # Compute score
        cover_count, current_score = _compute_score_and_cover(residues)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            best_cover = cover_count.copy()
            
    if best_residues is None:
        # Fallback: pure greedy
        residues = [0] * P_COUNT
        residues = _greedy_fill(residues, 0, start_time + 5.0)
        best_cover, best_score = _compute_score_and_cover(residues)
        best_residues = residues.copy()
        
    # Phase 2: Hill Climb
    # Budget: until 30s before deadline
    hc_deadline = DEADLINE - 30.0
    if time.time() < hc_deadline:
        residues, cover_count, current_score = _hill_climb(
            best_residues, best_cover, best_score, hc_deadline
        )
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            best_cover = cover_count.copy()
            
    # Phase 3: Random Restarts with Perturbation
    # Use remaining time
    # If we have more than 10 seconds left, do some restarts
    if time.time() < DEADLINE - 10.0:
        rng = np.random.default_rng()
        restart_deadline = DEADLINE - 5.0
        
        while time.time() < restart_deadline:
            # Perturb current best
            perturbed = best_residues.copy()
            num_perturb = rng.integers(3, 9)
            indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
            for i in indices:
                p = PRIMES[i]
                perturbed[i] = rng.integers(0, p)
                
            # Compute state
            p_cover, p_score = _compute_score_and_cover(perturbed)
            
            # Hill climb from perturbed
            # Give it a small budget, e.g., 5 seconds or until no improvement
            # To be safe, we limit the hill climb time per restart
            hc_time = min(5.0, restart_deadline - time.time())
            if hc_time <= 0:
                break
                
            r_res, r_cover, r_score = _hill_climb(
                perturbed, p_cover, p_score, time.time() + hc_time
            )
            
            if r_score > best_score:
                best_score = r_score
                best_residues = r_res.copy()
                best_cover = r_cover.copy()
                
    # Construct final set
    # Elements with cover_count == 0
    final_indices = np.nonzero(best_cover == 0)[0]
    
    return list(final_indices)

# Keep the signature
# The function is construct()

# To ensure it runs, we can add a simple check if __name__ == '__main__' for testing
# but the evaluator calls construct() directly.
