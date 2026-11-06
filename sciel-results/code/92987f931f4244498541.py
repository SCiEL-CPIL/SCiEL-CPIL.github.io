"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes):
    for p in primes:
        residues = set(pt % p for pt in points)
        if len(residues) == p:
            return False
    return True

def _greedy_construct(all_primes, N, seed=0):
    random.seed(seed)
    points = list(range(N + 1))
    
    # Precompute primes up to N+1
    # We process primes in a specific order. 
    # Standard: ascending. 
    # Randomized: shuffle the order? 
    # Note: The condition depends on len(points). 
    # If we process a large prime first, len(points) is large.
    # If we process small primes first, we remove points, len decreases.
    # Usually, small primes constrain more because p is small, so N/p is large.
    # Removing a class mod 2 removes ~N/2 points.
    # Removing a class mod large p removes ~N/p points.
    # To maximize size, we should probably handle large primes first? 
    # No, if we handle large primes first, we might remove few points.
    # Then small primes might force massive removals.
    # If we handle small primes first, we remove massive points.
    # The set of required primes shrinks as len(points) shrinks.
    
    # Let's try randomizing the order of primes.
    # But we must check p <= len(points).
    
    changed = True
    # We need to iterate until no more changes.
    # But if we shuffle, we might need multiple passes.
    
    # Optimization: 
    # The current deterministic method iterates primes 2, 3, 5...
    # If we shuffle, we might check a prime p, find it's not fully covered, skip it.
    # Then later, after removing points for other primes, it might become fully covered?
    # No, removing points only reduces the number of residues covered. 
    # If a residue class was empty, it stays empty.
    # If a residue class was non-empty, it might become empty.
    # So, "fully covered" status can only change from False to True? 
    # Wait. 
    # Initially, all classes are covered.
    # We remove a class. Now that class is empty.
    # For any other prime q, removing points might cause some class mod q to become empty.
    # So, a prime q that was NOT fully covered (had an empty class) might become FULLY covered?
    # No. If a class is empty, it stays empty unless we ADD points. We only remove.
    # So if a prime is not fully covered, it will never become fully covered.
    # Therefore, we only need to check primes that ARE fully covered.
    # Once we remove a class for p, p is no longer fully covered.
    # So we can process each prime at most once?
    # Not exactly. 
    # If we process p, and it's fully covered, we remove a class.
    # This removal might cause a DIFFERENT prime q to become fully covered?
    # No, q was already fully covered or not.
    # If q was fully covered, we need to handle it.
    # If q was NOT fully covered, it remains NOT fully covered.
    # So, we just need to ensure that for every prime p <= current_len, it is NOT fully covered.
    # We can iterate through primes. If p is fully covered, fix it.
    # Since fixing p only removes points, it cannot make any other prime q (that was not fully covered) become fully covered.
    # It CAN make a prime q (that was fully covered) remain fully covered (we still need to fix it).
    # So, we can simply iterate through all primes <= N+1.
    # For each p, if p <= current_len and p is fully covered, remove a class.
    # We don't need to loop until no change. One pass is sufficient?
    # Let's verify.
    # Start: All primes fully covered.
    # Pass 1:
    #   p=2. Fully covered? Yes. Remove class. Now 2 is NOT fully covered.
    #   p=3. Fully covered? Maybe. If yes, remove class.
    #   ...
    #   p=k. Fully covered? 
    #   The removal for p=2 might have affected the counts for p=3.
    #   But p=3 was fully covered before. It is still fully covered or maybe some class became empty?
    #   If some class for p=3 became empty due to p=2 removal, then p=3 is NOT fully covered. We skip it.
    #   If no class for p=3 became empty, p=3 is still fully covered. We remove a class.
    #   So, in a single forward pass, we handle each prime.
    #   Do we need a second pass?
    #   Suppose p=2 removal made p=3 NOT fully covered. We skipped p=3.
    #   Later, p=5 removal makes p=3... wait, removing points cannot make a non-empty class empty? 
    #   Yes it can.
    #   But if p=3 was NOT fully covered, it had an empty class.
    #   Removing more points cannot fill that empty class.
    #   So p=3 remains NOT fully covered.
    #   So one pass is sufficient to ensure admissibility, PROVIDED we check all primes.
    #   Wait, the condition is p <= len(points).
    #   As we remove points, len(points) decreases.
    #   So some large primes might drop out of the requirement.
    #   So we just need to check p <= current_len.
    
    # Algorithm:
    # 1. points = [0..N]
    # 2. For p in primes (sorted? or random?):
    #      if p > len(points): break (if sorted) or continue
    #      if p is fully covered:
    #          remove least populated (or random) class
    
    # Does the order matter?
    # If we remove a class for p=2, we lose ~N/2 points.
    # If we remove a class for p=large first, we lose ~N/large points.
    # To maximize size, we want to lose as few points as possible.
    # Removing a class for large p is "cheaper" in terms of points lost.
    # However, we MUST remove a class for EVERY prime p <= L.
    # So we have to pay the cost for all of them.
    # The cost is roughly sum(N/p) for all p <= L?
    # No, the removals interact.
    
    # Let's try sorting primes in DESCENDING order.
    # Handle large primes first. They require removing a small fraction of points.
    # Then handle small primes.
    
    # Let's implement randomized order and tie-breaking.
    
    # We'll use a list of primes.
    # We'll shuffle the primes? 
    # If we shuffle, we might process a small prime late.
    # By then, len(points) might be smaller, so the small prime might not be required?
    # No, small primes are always required if len(points) >= p.
    # Since len(points) starts at 5001, all primes up to 5000 are initially required.
    # As we remove points, len decreases.
    # If we process large primes first, we remove few points. len stays high.
    # Then we process small primes, remove many points.
    # If we process small primes first, we remove many points. len drops.
    # Maybe large primes are no longer required?
    # Example: N=10. Primes 2,3,5,7.
    # Len=11.
    # Process 2: remove ~5. Len ~6.
    # Process 3: remove ~2. Len ~4.
    # Process 5: 5 > 4, skip.
    # Process 7: 7 > 4, skip.
    # Total removed ~7.
    
    # If we process 7 first:
    # Len=11.
    # Process 7: remove ~1. Len ~10.
    # Process 5: remove ~2. Len ~8.
    # Process 3: remove ~2. Len ~6.
    # Process 2: remove ~3. Len ~3.
    # Total removed ~8.
    
    # It seems processing small primes first might allow skipping larger primes later?
    # In the first case, we skipped 5 and 7.
    # In the second case, we processed 7, 5, 3, 2.
    # So small-first is better?
    
    # Let's try both orders and random orders.
    
    primes_list = [p for p in all_primes if p <= N+1]
    
    # Try different strategies
    best_points = []
    best_len = -1
    
    # Strategy 1: Ascending primes, least populated tie-break
    # Strategy 2: Descending primes, least populated tie-break
    # Strategy 3: Random primes, random tie-break
    
    # Given time limit, we can run many random restarts.
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Pre-compute residues for speed?
    # points is a list.
    
    # Fast check for "fully covered"
    # For a given p, we need to know if all 0..p-1 are present.
    
    # We'll run the greedy loop with different configurations.
    
    configs = []
    # Ascending
    configs.append(('asc', 0))
    # Descending
    configs.append(('desc', 0))
    
    # Random restarts
    num_randoms = 100 # Adjust based on time
    
    for i in range(num_randoms):
        if time.time() > deadline:
            break
        configs.append(('rand', i))
        
    for order_type, seed in configs:
        if time.time() > deadline:
            break
            
        random.seed(seed)
        pts = list(range(N + 1))
        
        # Prepare primes list
        if order_type == 'asc':
            ps = [p for p in all_primes if p <= N+1]
        elif order_type == 'desc':
            ps = [p for p in all_primes if p <= N+1][::-1]
        else:
            ps = [p for p in all_primes if p <= N+1]
            random.shuffle(ps)
        
        # Process
        for p in ps:
            if p > len(pts):
                continue
                
            # Check if fully covered
            # Use numpy for speed
            residues = np.array([pt % p for pt in pts])
            counts = np.bincount(residues, minlength=p)
            
            if counts.min() > 0:
                # Fully covered
                # Choose class to remove
                if order_type == 'rand' and random.random() < 0.5:
                    # Random class among those with min count? Or any?
                    # Let's pick random among min count
                    min_val = counts.min()
                    candidates = np.where(counts == min_val)[0]
                    min_class = candidates[random.randint(0, len(candidates)-1)]
                else:
                    min_class = np.argmin(counts)
                    
                # Remove
                # Filter list
                # pts = [pt for pt in pts if pt % p != min_class]
                # Vectorized removal?
                mask = (np.array(pts) % p) != min_class
                pts = [pt for pt, m in zip(pts, mask) if m]
                
        # Check length
        if len(pts) > best_len:
            best_len = len(pts)
            best_points = pts[:]
            
    # Final check
    if not best_points:
        best_points = list(range(N+1)) # Fallback, though likely invalid
        
    # Ensure admissibility
    # The greedy should have done it, but verify.
    # If not admissible, remove classes.
    final_primes = [p for p in all_primes if p <= len(best_points)]
    for p in final_primes:
        residues = set(pt % p for pt in best_points)
        if len(residues) == p:
            counts = np.zeros(p, dtype=int)
            for pt in best_points:
                counts[pt % p] += 1
            min_class = np.argmin(counts)
            best_points = [pt for pt in best_points if pt % p != min_class]
            
    return best_points

def construct():
    return _greedy_construct(_primes(N+1), N)
# EVOLVE-BLOCK-END
