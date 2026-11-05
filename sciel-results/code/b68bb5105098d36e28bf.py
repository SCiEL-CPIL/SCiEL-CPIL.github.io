import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_counts(residues, primes, max_L):
    """Compute the coverage count array for a given set of residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
    return counts

def _find_L(counts, max_L):
    """Find the largest L such that every integer 1..L is covered at least twice."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def _greedy_construct(primes, max_L, seed=0):
    """
    Construct an initial solution using a greedy forward approach.
    At each step, for prime p_i, choose the residue r that maximizes the 
    'deficiency' reduction in the current window, or simply maximizes coverage
    in the uncovered parts.
    
    To ensure diversity, we add some randomness to the tie-breaking or the 
    initial selection window.
    """
    np.random.seed(seed)
    residues = [0] * len(primes)
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # We process primes in increasing order.
    # For each prime, we want to cover as many "deficient" spots as possible.
    # A spot is deficient if its count is < 2.
    
    for i, p in enumerate(primes):
        # Determine the current "deficiency" map
        # We only care about the first max_L integers.
        # To speed up, we can focus on the region where the current L is likely to be.
        # But for simplicity and robustness, we evaluate over a window.
        # Let's try all residues for small p, and sample for large p.
        
        if p <= 50:
            candidates = list(range(p))
        else:
            # Sample residues
            candidates = list(np.random.choice(p, size=min(p, 50), replace=False))
        
        best_r = 0
        best_score = -1
        
        # We want to maximize the number of integers that go from count < 2 to count >= 2,
        # or more generally, increase the minimum count in the prefix.
        # A good proxy: maximize the sum of min(count[x]+1, 2) - min(count[x], 2) for x in 1..max_L
        # This is equivalent to counting how many positions we "complete" to 2.
        
        # Precompute the base counts (without this prime)
        base_counts = counts.copy()
        
        for r in candidates:
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                continue
            
            pos = np.arange(start, max_L + 1, p)
            if len(pos) == 0:
                continue
            
            # Calculate how many of these positions are currently < 2
            # and how many are currently >= 2
            # If we add 1 to these positions:
            # - If count was 0, it becomes 1 (no improvement in "covered twice" status)
            # - If count was 1, it becomes 2 (improvement: now covered twice)
            # - If count was >= 2, it becomes >= 3 (no improvement in "covered twice" status)
            
            # So the gain is the number of positions in pos where base_counts == 1.
            gain = np.sum(base_counts[pos] == 1)
            
            # Also, we might want to prioritize covering the early part of the array.
            # Let's add a small bias towards covering early integers.
            # We can weight the gain by 1/(x+1) or similar.
            # For speed, let's just use the count of completions.
            
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[i] = best_r
        # Update counts
        if best_r == 0:
            start = p
        else:
            start = best_r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
                
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Generate multiple diverse starting points using greedy construction
    # The greedy method with different seeds should produce different local optima.
    best_residues = None
    best_L = 0
    
    # We will run a few greedy constructions and then refine the best one with SA.
    # Or we can run SA from multiple starts.
    # Given the time limit, let's do:
    # 1. Generate 5 greedy starts.
    # 2. Evaluate them.
    # 3. Pick the best one.
    # 4. Run SA from that best one.
    
    greedy_starts = []
    for s in range(5):
        if (time.time() - start_time) > deadline - 10:
            break
        res = _greedy_construct(primes, max_L, seed=s)
        counts = _compute_counts(res, primes, max_L)
        L = _find_L(counts, max_L)
        greedy_starts.append((L, res, counts))
        
    if not greedy_starts:
        # Fallback to simple heuristic if greedy takes too long (unlikely)
        initial_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
        counts = _compute_counts(initial_residues, primes, max_L)
        L = _find_L(counts, max_L)
        best_residues = initial_residues
        best_L = L
    else:
        # Sort by L descending
        greedy_starts.sort(key=lambda x: x[0], reverse=True)
        best_L, best_residues, counts = greedy_starts[0]
        
    # Now run Simulated Annealing from the best greedy start
    # We will use the precomputed counts for the current best.
    current_residues = best_residues[:]
    current_counts = counts
    current_L = best_L
    
    # SA Parameters
    # Temperature schedule: exponential decay
    # We need to fit the SA loop within the remaining time.
    remaining_time = deadline - (time.time() - start_time)
    
    # Estimate time per SA step.
    # A step involves:
    # 1. Pick a random prime i.
    # 2. Pick a random new residue r.
    # 3. Update counts (vectorized subtraction/addition).
    # 4. Calculate new L (vectorized search).
    # 5. Decide accept/reject.
    
    # Let's benchmark one step to estimate throughput.
    t0 = time.time()
    steps_done = 0
    temp = 1000.0
    temp_min = 0.1
    cooling_rate = 0.999
    
    # We will run SA until time runs out.
    # To avoid overhead, we'll do batches of steps if possible, but numpy ops are fast.
    
    rng = np.random.default_rng(42)
    
    while (time.time() - start_time) < deadline - 2:
        # Pick a random prime
        i = rng.integers(0, n_primes)
        p = primes[i]
        old_r = current_residues[i]
        
        # Pick a new random residue
        new_r = rng.integers(0, p)
        
        if new_r == old_r:
            continue
            
        # Calculate delta in counts
        # Remove old coverage
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        
        if old_start <= max_L:
            old_pos = np.arange(old_start, max_L + 1, p)
            if len(old_pos) > 0:
                current_counts[old_pos] -= 1
        else:
            old_pos = np.array([], dtype=np.int32)
            
        # Add new coverage
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
            
        if new_start <= max_L:
            new_pos = np.arange(new_start, max_L + 1, p)
            if len(new_pos) > 0:
                current_counts[new_pos] += 1
        else:
            new_pos = np.array([], dtype=np.int32)
            
        # Calculate new L
        new_L = _find_L(current_counts, max_L)
        
        # Acceptance probability
        delta = new_L - current_L
        if delta >= 0:
            accept = True
        else:
            # Metropolis criterion
            # exp(delta / temp)
            # delta is negative.
            # If temp is small, this is close to 0.
            # If delta is large negative, this is very small.
            # To avoid underflow, we can use log space or just check if delta > -temp * 10
            if delta > -10 * temp:
                prob = np.exp(delta / temp)
                accept = rng.random() < prob
            else:
                accept = False
                
        if not accept:
            # Revert
            if len(old_pos) > 0:
                current_counts[old_pos] += 1
            if len(new_pos) > 0:
                current_counts[new_pos] -= 1
            # current_residues[i] remains old_r
        else:
            # Accept
            current_residues[i] = new_r
            current_L = new_L
            
            if current_L > best_L:
                best_L = current_L
                best_residues = current_residues[:]
                
        # Cool down
        temp *= cooling_rate
        if temp < temp_min:
            # Reheat or stop?
            # If we stop, we might miss improvements.
            # Let's reheat to a moderate value to explore again.
            # Or just keep going with a low temp (effectively greedy).
            # Reheating can help escape local minima.
            temp = 100.0 
            cooling_rate = 0.9995 # slower cooling after reheat?
            
        steps_done += 1
        if steps_done % 1000 == 0:
            # Check time
            if (time.time() - start_time) > deadline - 2:
                break

    return best_residues
