"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Precompute primes up to N
    primes = _primes(N)
    primes_list = primes.tolist()
    
    # 2. Strategy: Randomized Greedy Removal
    # We start with the full set [0, N].
    # We must remove residue classes for primes p <= |S| such that not all residues are covered.
    # The current deterministic approach removes the *least* populated residue class.
    # This is a local greedy choice.
    # A different strategy: Randomly select a residue class to remove (among those that are "safe" to remove? 
    # Actually, we MUST remove *some* residue if all are covered. We can choose WHICH one to remove.
    # Removing a specific residue class r mod p removes all x in S such that x % p == r.
    # The "cost" is the number of elements removed.
    # To maximize final size, we want to remove as few elements as possible at each step.
    # However, removing a small class now might prevent the need to remove a large class later?
    # Or it might force us to remove more later?
    
    # Let's try a randomized approach with multiple restarts.
    # In each restart, we randomly pick one of the available residue classes to remove (if all are covered).
    # Or better: we pick the residue class with the minimum count, but add some randomness (e.g. softmax over inverse counts).
    # Or simply: pick randomly among the residues.
    
    # Let's try a "Simulated Annealing" like approach or just multiple random restarts of the greedy process.
    # Since the process is deterministic given the choices, we can sample different choices.
    
    best_size = 0
    best_mask = None
    
    num_restarts = 100  # We can do many restarts if they are fast
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = np.ones(N + 1, dtype=bool)
        
        # Process primes in increasing order
        # We need to check primes p <= |S|.
        # Initially |S| = N+1.
        # As we remove elements, |S| decreases.
        
        # To speed up, we can precompute the counts for each prime for the full set?
        # No, counts change as we remove.
        
        # Optimization:
        # For each prime p, we maintain the count of each residue.
        # Initially, for p, the counts are roughly (N+1)//p.
        # When we remove a residue class r mod p', it affects counts for all primes p.
        # This is expensive to update globally.
        
        # Alternative: Recompute counts on the fly using vectorized ops, but only for the primes we need to check.
        # The number of primes we need to check is small (up to |S|).
        # But we iterate through primes in order.
        
        # Let's stick to the vectorized recompute but make the choice random.
        
        current_size = N + 1
        
        # We iterate through primes.
        # If p > current_size, we break.
        
        for p in primes_list:
            if p > current_size:
                break
                
            # Check coverage
            active_points = np.where(mask)[0]
            residues = active_points % p
            counts = np.bincount(residues, minlength=p)
            
            if np.all(counts > 0):
                # All residues covered. Must remove one.
                # Strategy: Randomly pick a residue to remove.
                # To bias towards better choices, we could pick from the ones with lowest count?
                # Let's try: Pick uniformly at random from all residues.
                # Or pick the one with the minimum count, but with some probability pick another?
                
                # Let's try a "Soft" greedy: 
                # Probability proportional to 1/count? Or just uniform?
                # Uniform might be too random and remove too many.
                # Let's try: With 50% probability, remove the min-count residue.
                # With 50% probability, remove a random residue.
                
                if np.random.rand() < 0.5:
                    # Greedy step
                    min_residue = np.argmin(counts)
                else:
                    # Random step
                    min_residue = np.random.randint(0, p)
                
                # Remove points in this residue class
                to_remove = (np.arange(N + 1) % p == min_residue) & mask
                removed_count = np.sum(to_remove)
                mask[to_remove] = False
                current_size -= removed_count
                
        # End of removal phase for this restart
        
        # Check if we have a valid set
        # The removal phase ensures validity for all p <= final_size?
        # Wait. The condition is: for every prime p <= |S|, some residue is missed.
        # My loop breaks when p > current_size.
        # So for all p <= current_size, we ensured that at the time we processed p, we removed a residue if needed.
        # But removing points for a *later* prime (larger p) might *restore* coverage for an *earlier* prime?
        # No, removing points only reduces the set. It cannot add new residues.
        # So if a residue was missed for p at step p, it remains missed.
        # So the set is valid.
        
        # Now, Expansion Phase
        # Try to add back removed points?
        # This is the hard part.
        # Let's try a simple local search: 
        # For each point x not in mask, try to add it.
        # Check if it violates any prime p <= |S|+1.
        
        # To make this fast, we maintain the "empty residues" for each prime.
        # This is expensive to initialize.
        # Given the time budget, maybe we skip expansion for the random restarts?
        # Or do a very limited expansion?
        
        # Let's check the size.
        if current_size > best_size:
            best_size = current_size
            best_mask = mask.copy()
            
    # If we have time, do a more careful expansion on the best mask found so far.
    # Or, better: modify the removal to be less aggressive?
    
    # Actually, the "Expansion" in the previous program was key.
    # Can we integrate expansion into the restarts?
    # Maybe not, it's too slow.
    
    # Let's try a different strategy for the main loop:
    # Instead of removing *all* points in a residue class, can we remove *fewer*?
    # No, the condition is "some residue class is missed".
    # If we remove *any* point from residue class r, does that mean r is missed?
    # No! "Missed" means NO points in S have that residue.
    # So we MUST remove ALL points in that residue class.
    # So the choice is binary: remove the whole class or not (if it's already empty).
    
    # So the search space is: for each prime p, which residue to kill?
    # This is a combinatorial problem.
    
    # The randomized greedy above samples one path.
    # The expansion phase tries to fix over-removal.
    
    # Let's implement a more robust expansion on the best candidate.
    
    if best_mask is None:
        # Fallback
        mask = np.ones(N + 1, dtype=bool)
        for p in primes_list:
            if p > np.sum(mask):
                break
            active_points = np.where(mask)[0]
            residues = active_points % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                min_residue = np.argmin(counts)
                to_remove = (np.arange(N + 1) % p == min_residue) & mask
                mask[to_remove] = False
        best_mask = mask
        best_size = np.sum(mask)
        
    # Expansion Phase on best_mask
    mask = best_mask
    current_size = best_size
    
    # Precompute empty residues for relevant primes
    # Relevant primes are p <= current_size + 1 (since we might add 1 point)
    # Actually, we might add multiple points, so p <= N+1.
    
    # To be efficient, we only care about primes p <= current_size + 1 initially.
    # As we add points, the size grows, so more primes become relevant.
    
    # Let's maintain the state.
    # This is O(N * pi(N)) to initialize.
    # pi(5000) is ~ 670.
    # 5000 * 670 is 3.3 million ops. Very fast in NumPy.
    
    max_prime_limit = N + 1
    relevant_primes = [p for p in primes_list if p <= max_prime_limit]
    
    empty_flags = {}
    empty_counts = {}
    
    # Initialize
    active_points = np.where(mask)[0]
    for p in relevant_primes:
        if len(active_points) == 0:
            empty_flags[p] = np.ones(p, dtype=bool)
            empty_counts[p] = p
        else:
            residues = active_points % p
            present = np.zeros(p, dtype=bool)
            present[residues] = True
            empty_flags[p] = ~present
            empty_counts[p] = np.sum(empty_flags[p])
            
    candidates = np.where(~mask)[0]
    
    # Greedy add-back
    # Sort candidates? 
    # Maybe sort by "potential"? 
    # A point x is "good" if it falls into many empty classes? No, that makes it *worse* (more likely to fill the last hole).
    # A point x is "good" if it falls into *non-empty* classes for most primes?
    # If x % p is already present, adding x doesn't change the empty set for p.
    # So x is "safe" for p if x % p is present, OR if x % p is empty but there are >= 2 empty classes.
    # The "most safe" points are those that are present in many classes.
    # Let's score candidates by the number of primes p <= current_size+1 where x % p is present.
    # Higher score = safer.
    
    # Computing this score for all candidates is expensive: N * pi(N).
    # 5000 * 670 = 3.3M. Okay.
    
    scores = np.zeros(len(candidates))
    for i, x in enumerate(candidates):
        score = 0
        for p in relevant_primes:
            if p > current_size + 1:
                break
            r = x % p
            if not empty_flags[p][r]:
                score += 1
        scores[i] = score
        
    # Sort candidates by score descending
    order = np.argsort(-scores)
    
    for idx in order:
        if time.time() > deadline:
            break
            
        x = candidates[idx]
        new_size = current_size + 1
        
        safe = True
        for p in relevant_primes:
            if p > new_size:
                break
            r = x % p
            if empty_flags[p][r]:
                if empty_counts[p] == 1:
                    safe = False
                    break
                    
        if safe:
            mask[x] = True
            current_size += 1
            
            # Update flags
            for p in relevant_primes:
                if p > current_size: # Note: size increased, so p <= current_size might be new
                    break
                r = x % p
                if empty_flags[p][r]:
                    empty_flags[p][r] = False
                    empty_counts[p] -= 1
                    
    result = np.where(mask)[0].tolist()
    return result
# EVOLVE-BLOCK-END
