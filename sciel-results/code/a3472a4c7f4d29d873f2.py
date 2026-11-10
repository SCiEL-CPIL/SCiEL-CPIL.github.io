"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_fast(A):
    """Optimized score computation using local variables and direct set building."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    lss = len(ss)
    lps = len(ps)
    return lss if lss > lps else lps

def _gen_smooth_numbers(limit, max_prime=5):
    """Generate smooth numbers up to limit using primes <= max_prime."""
    primes = []
    d = 2
    while d <= max_prime:
        is_p = True
        for p in primes:
            if p * p > d:
                break
            if d % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(d)
        d += 1
    
    smooth = set([1])
    for p in primes:
        current = list(smooth)
        mult = 1
        while mult * p <= limit:
            for s in current:
                val = s * mult * p
                if val <= limit:
                    smooth.add(val)
            mult *= p
    return sorted(smooth)

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We target finishing well before.
    # Use 85s as hard deadline for search, leaving 5s for final checks/return.
    deadline = start_time + 85.0
    
    n = 20
    maximum = 10000
    
    # 1. Generate Candidate Pools
    # Pool A: 5-smooth numbers (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, ...)
    smooth_5 = _gen_smooth_numbers(maximum, max_prime=5)
    
    # Pool B: 7-smooth numbers (adds 7)
    smooth_7 = _gen_smooth_numbers(maximum, max_prime=7)
    
    # Pool C: Small integers (1..100)
    small_ints = list(range(1, 101))
    
    # Combine into a master pool, ensuring uniqueness and sorting
    master_pool = sorted(set(smooth_5 + smooth_7 + small_ints))
    
    # Filter out values > maximum (already done in gen, but double check)
    master_pool = [x for x in master_pool if x <= maximum]
    
    # 2. Define Initial Candidate Sets
    candidates = []
    
    # Candidate 1: First 20 5-smooth numbers
    if len(smooth_5) >= n:
        candidates.append(sorted(smooth_5[:n]))
    
    # Candidate 2: First 20 7-smooth numbers
    if len(smooth_7) >= n:
        candidates.append(sorted(smooth_7[:n]))
    
    # Candidate 3: Geometric progression base (1, 2, 4, ..., 2^19 is too big, so 1..2^13)
    # 2^13 = 8192. 14 elements. Need 6 more.
    geo_base = [2**i for i in range(14)]
    # Add small odd numbers to fill
    extras = [3, 5, 7, 9, 11, 13]
    cand_geo = sorted(set(geo_base + extras))
    if len(cand_geo) >= n:
        candidates.append(cand_geo[:n])
    
    # Candidate 4: Mix of small smooth and slightly larger smooth
    # Take 10 small, 10 from middle range
    if len(smooth_5) >= 40:
        cand_mix = smooth_5[:10] + smooth_5[15:25]
        cand_mix = sorted(set(cand_mix))
        if len(cand_mix) >= n:
            candidates.append(cand_mix[:n])

    # Ensure at least one valid candidate
    if not candidates:
        candidates.append(list(range(1, n+1)))
    
    # 3. Evaluate Initial Candidates
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        if time.time() > deadline - 10: # Leave time for search
            break
        if len(A) != n or len(set(A)) != n:
            continue
        if any(x <= 0 or x > maximum for x in A):
            continue
        
        sc = _score_fast(A)
        if sc < best_score:
            best_score = sc
            best_A = A[:]
    
    if best_A is None:
        best_A = list(range(1, n+1))
        best_score = _score_fast(best_A)
    
    # 4. Local Search with Multiplicative Perturbation Strategy
    # The directive suggests: Model A as {b^e_i * k_i}. 
    # In practice, for local search, we interpret this as:
    # - Primary moves: Replace an element with a nearby smooth number (preserving structure).
    # - Secondary moves: "Perturb" by multiplying/dividing by small primes (2,3,5,7) 
    #   if the result is an integer in range and distinct. This explores the "k_i" space.
    
    rng = random.Random(12345)
    A = best_A[:]
    current_score = best_score
    
    # Precompute set for O(1) membership checks
    A_set = set(A)
    
    # Define specific perturbation operations
    def get_perturbations(val, A_set):
        """Generate candidate values near val via multiplication/division by small primes."""
        cands = set()
        # Multiply by small primes
        for p in [2, 3, 5, 7, 11]:
            new_val = val * p
            if new_val <= maximum and new_val not in A_set:
                cands.add(new_val)
            # Divide by small primes (if divisible)
            if val % p == 0:
                new_val = val // p
                if new_val >= 1 and new_val not in A_set:
                    cands.add(new_val)
        # Also add +/- 1, +/- 2 small steps for additive fine-tuning
        for delta in [-2, -1, 1, 2]:
            new_val = val + delta
            if 1 <= new_val <= maximum and new_val not in A_set:
                cands.add(new_val)
        return list(cands)

    improved = True
    search_start = time.time()
    
    while improved and time.time() < deadline - 2:
        improved = False
        # Randomize the order of indices to avoid bias
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2:
                break
            
            old_val = A[i]
            A_set.remove(old_val)
            
            # 1. Try Perturbations (Multiplicative structure)
            perturbations = get_perturbations(old_val, A_set)
            rng.shuffle(perturbations)
            
            # 2. Try Random Smooth Numbers from Master Pool
            # Sample a subset of the pool to keep it fast
            # We prioritize numbers that are "close" in log-space to old_val to maintain balance
            # But for simplicity, we just sample from the whole pool or a sorted slice.
            # Let's try a mix: 50% perturbations, 50% random from pool
            
            # To speed up, we limit the number of random pool tries per step
            num_pool_tries = 20
            
            best_candidate_val = None
            best_candidate_score = current_score
            
            # Test perturbations
            for new_val in perturbations:
                A[i] = new_val
                A_set.add(new_val)
                sc = _score_fast(A)
                if sc < best_candidate_score:
                    best_candidate_score = sc
                    best_candidate_val = new_val
                A[i] = old_val
                A_set.remove(new_val)
                A_set.add(old_val)
            
            # Test random pool elements
            # Select random indices from master_pool
            # We can bias towards values near the current value to avoid huge jumps that might break structure
            # Find index of old_val in master_pool (approx)
            # For simplicity, just random choice from pool
            for _ in range(num_pool_tries):
                new_val = rng.choice(master_pool)
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                A_set.add(new_val)
                sc = _score_fast(A)
                if sc < best_candidate_score:
                    best_candidate_score = sc
                    best_candidate_val = new_val
                A[i] = old_val
                A_set.remove(new_val)
                A_set.add(old_val)
            
            if best_candidate_val is not None:
                # Apply the best move found in this iteration
                A[i] = best_candidate_val
                A_set.add(best_candidate_val)
                current_score = best_candidate_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
                improved = True
                # Break out of index loop to restart with new state? 
                # Or continue? Usually restarting index order is good.
                break
            else:
                # Re-add old val if it was removed and not replaced
                # In the loop above, we restored A[i] and A_set in each test.
                # So A[i] is still old_val and A_set contains old_val.
                pass
        
        # If no improvement in a full pass, try a more aggressive random restart of local moves
        if not improved:
            # Try a few random single-step moves that might accept worse scores temporarily?
            # No, we want to minimize. Let's just try more random perturbations without strict improvement requirement?
            # Actually, if no improvement is found, we might be stuck in a local minimum.
            # Try "Simulated Annealing" style: accept worse moves with small probability?
            # Given the strict time limit and discrete nature, let's just try a few random restarts of the local search
            # by perturbing the current best solution randomly.
            
            # Generate a perturbed version of best_A
            temp_A = best_A[:]
            temp_score = best_score
            for _ in range(10):
                idx = rng.randint(0, n-1)
                old_v = temp_A[idx]
                temp_A_set = set(temp_A)
                temp_A_set.remove(old_v)
                # Pick a random new value from pool
                new_v = rng.choice(master_pool)
                if new_v in temp_A_set:
                    continue
                temp_A[idx] = new_v
                temp_A_set.add(new_v)
                new_sc = _score_fast(temp_A)
                if new_sc < temp_score:
                    temp_score = new_sc
                    # If we found a better one, update global best
                    if new_sc < best_score:
                        best_score = new_sc
                        best_A = temp_A[:]
                        A = temp_A[:]
                        A_set = set(A)
                        current_score = best_score
                        improved = True
                        break
                else:
                    # Revert
                    temp_A[idx] = old_v
                    temp_A_set.remove(new_v)
                    temp_A_set.add(old_v)
            
            if not improved:
                # If still no improvement, we might exit early to save time, 
                # or continue with smaller steps. 
                # Let's break to avoid wasting time if stuck.
                break

    # 5. Final Validation
    if not _valid(best_A, n, maximum):
        # Fallback to a known valid set
        best_A = list(range(1, n+1))
    
    return best_A

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True
# EVOLVE-BLOCK-END
