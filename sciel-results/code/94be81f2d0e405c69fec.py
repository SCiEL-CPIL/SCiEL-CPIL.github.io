import time
import random

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val=10000):
    """Generate smooth numbers with given prime bases."""
    result = set()
    def rec(idx, current):
        if current > max_val:
            return
        result.add(current)
        if idx < len(primes):
            val = current
            while val * primes[idx] <= max_val:
                val *= primes[idx]
                rec(idx + 1, val)
    rec(0, 1)
    return sorted(result)

def local_search(A, time_limit, rng, candidate_pool=None):
    """
    Perform local search to minimize the score.
    Tries to replace elements in A with other integers to improve the score.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    if candidate_pool is None:
        # Pool of candidates for replacement
        # We generate a mix of small integers and some smooth numbers
        # to provide diverse replacement options
        candidate_pool = list(range(1, 101)) # Small integers
        # Add some smooth numbers
        for i in range(1, 15):
            for j in range(1, 15):
                val = (2 ** i) * (3 ** j)
                if val <= 10000:
                    candidate_pool.append(val)
        candidate_pool = list(set(candidate_pool))
        rng.shuffle(candidate_pool)
    
    # Ensure A is sorted for consistent indexing in local search
    A = sorted(A)
    best_A = sorted(best_A)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try swapping each element in A with a candidate from the pool
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            # Try a subset of candidates to keep it fast, or all if fast enough
            # n=20, pool size ~200. 20*200 = 4000 swaps.
            # Each swap involves O(n^2) set updates. 400 * 4000 = 1.6M ops.
            # This is fast in Python if implemented carefully.
            
            for new_val in candidate_pool:
                if new_val == old_val or new_val in A:
                    continue
                
                # Create new set
                new_A = A.copy()
                new_A[i] = new_val
                new_A.sort()
                
                # Quick check: distinctness is ensured by 'new_val in A' check above
                # Calculate score
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    # Update best
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    # Restart search from this new point? 
                    # Or continue? Usually restart or continue. 
                    # Let's continue to see if we can improve further immediately.
                    # But we must be careful not to loop. 
                    # Since score strictly decreases, it will terminate.
                    break # Move to next i? Or restart i loop?
                    # Breaking inner loop and moving to next i is a standard greedy step.
        
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    rng = random.Random(42)
    
    candidates = []
    
    # Candidate 1: Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))
    
    # Candidate 2: Geometric Progression 1, 2, 4, ...
    # 2^19 is too big? 2^19 = 524288. Max is 10000.
    # So we can only do 2^0 to 2^13 (16 elements).
    # Let's adjust: 1, 2, 4, ..., 2^13, and then some others?
    # Or just a geometric progression with ratio close to 1?
    # Let's stick to valid ones.
    gp = [2**i for i in range(14)] # 16 elements
    # Add 4 more small distinct integers not in gp
    for x in range(1, 100):
        if x not in gp and len(gp) < 20:
            gp.append(x)
    candidates.append(gp)
    
    # Candidate 3: 2,3-smooth numbers
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    # Pick 20 smallest
    candidates.append(smooth[:20])
    
    # Candidate 4: Random set of small integers
    rand_set = random.sample(range(1, 101), 20)
    candidates.append(rand_set)
    
    # Candidate 5: Mixed structure - low AP + high sparse
    mixed = list(range(1, 11)) + [12, 14, 16, 20, 25, 30, 40, 50, 60, 80]
    candidates.append(mixed)
    
    # Candidate 6: 2,3,5-smooth numbers (inspired by smooth number generation)
    smooth_235 = _generate_smooth([2, 3, 5])
    candidates.append(smooth_235[:20])
    
    # Candidate 7: 2,3,5,7-smooth numbers
    smooth_2357 = _generate_smooth([2, 3, 5, 7])
    candidates.append(smooth_2357[:20])
    
    # Candidate 8: Geometric progression with ratio ~1.3
    geo = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.3))
        if val > 10000:
            break
        if val not in geo:
            geo.append(val)
    if len(geo) >= 20:
        candidates.append(geo[:20])
    
    # Candidate 9: Geometric progression with ratio ~1.5
    geo2 = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.5))
        if val > 10000:
            break
        if val not in geo2:
            geo2.append(val)
    if len(geo2) >= 20:
        candidates.append(geo2[:20])
    
    # Candidate 10: Mix: small numbers + powers of 2
    mix = list(range(1, 11)) + [2**i for i in range(4, 14)]
    mix = sorted(set(x for x in mix if x <= 10000))
    if len(mix) >= 20:
        candidates.append(mix[:20])
    
    # Candidate 11: Random sample from smooth pool
    if len(smooth_235) >= 20:
        candidates.append(sorted(random.sample(smooth_235, 20)))
    
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        if time.time() > deadline:
            break
            
        # Ensure validity
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        
        # Run local search on this candidate
        # Allocate remaining time / number of remaining candidates
        # But local search has its own time limit.
        # Let's give each candidate a fixed time slice, e.g., 10s, but capped by total deadline.
        time_remaining = deadline - time.time()
        if time_remaining < 1:
            break
            
        # Use a portion of the remaining time for this candidate
        # If it's the first, maybe give it more time?
        # Let's just use a fixed 10s per candidate if enough time, else less.
        alloc_time = min(10.0, time_remaining * 0.8)
        
        improved_A, score = local_search(A, alloc_time, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    if best_A is None:
        # Fallback
        best_A = list(range(1, 21))
        
    return best_A
