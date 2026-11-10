import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_candidate_pool():
    """
    Generate a pool of candidate integers likely to be in optimal sets.
    Focus on small integers (good for sumsets) and smooth numbers (good for product sets).
    """
    candidates = set()
    
    # 1. Small integers (1-300)
    # These are crucial for keeping sums small and allowing many collisions.
    candidates.update(range(1, 301))
    
    # 2. 2, 3, 5, 7, 11-smooth numbers
    # Numbers with only small prime factors tend to have more multiplicative collisions.
    primes = [2, 3, 5, 7, 11]
    max_val = 10000
    
    def gen_smooth(primes, max_val, current=1, index=0):
        if current > max_val:
            return
        if index == len(primes):
            candidates.add(current)
            return
        p = primes[index]
        gen_smooth(primes, max_val, current, index + 1)
        val = current
        while val * p <= max_val:
            val *= p
            gen_smooth(primes, max_val, val, index)
            
    gen_smooth(primes, max_val)
    
    return sorted(candidates)

def local_search(A, time_limit, candidate_pool, rng):
    """
    Perform local search on the set A.
    Uses incremental evaluation of sumset/prodset size changes for speed.
    """
    A = list(A)
    A.sort()
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Pre-compute a smaller, high-priority subset of the candidate pool
    # Small numbers and very smooth numbers are the most likely improvements.
    # We limit the pool size to keep the inner loop fast.
    # Prioritize: 1-100, then smooth numbers up to 2000.
    priority_pool = [x for x in candidate_pool if x <= 2000]
    if len(priority_pool) > 1000:
        # Ensure we have a mix of small and smooth
        smalls = [x for x in priority_pool if x <= 100]
        smooths = [x for x in priority_pool if x > 100]
        # Shuffle smooths to avoid bias towards specific values in early iterations
        rng.shuffle(smooths)
        priority_pool = smalls + smooths[:1000 - len(smalls)]
        
    improved = True
    no_improve_count = 0
    
    # Maintain current sumset and prodset for incremental updates
    # Note: We store them as sets to calculate deltas accurately
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
            
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        A_set = set(A)
        
        # If we've had many iterations without improvement, we might be stuck.
        # We don't explicitly restart here to keep logic simple, but the 
        # incremental nature allows many fast evaluations.
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            best_new_val = None
            best_est_score = current_score
            best_new_sumset = None
            best_new_prodset = None
            
            # Iterate through candidate pool
            # To speed up, we can limit candidates to those 'close' to old_val 
            # or just iterate all. Pool is limited to ~1000 items.
            for new_val in priority_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                # Calculate delta for sumset and prodset
                # New elements added: new_val + A[j] and new_val * A[j] for j != i
                # Old elements removed: old_val + A[j] and old_val * A[j] for j != i
                # Also handle the i==i case: 2*val and val*val
                
                new_sum_elements = set()
                new_prod_elements = set()
                old_sum_elements = set()
                old_prod_elements = set()
                
                for j in range(n):
                    if j == i:
                        continue
                    aj = A[j]
                    new_sum_elements.add(new_val + aj)
                    new_prod_elements.add(new_val * aj)
                    old_sum_elements.add(old_val + aj)
                    old_prod_elements.add(old_val + aj)
                
                # Self terms
                new_sum_elements.add(2 * new_val)
                new_prod_elements.add(new_val * new_val)
                old_sum_elements.add(2 * old_val)
                old_prod_elements.add(old_val * old_val)
                
                # Calculate new sizes
                # Elements that are in new_set but not in old_set (net additions)
                # Elements that are in old_set but not in new_set (net removals)
                # Note: We must be careful with elements that appear multiple times.
                # However, since we are swapping one element, the "count" of a sum/prod
                # changing from 1 to 0 or 0 to 1 is what matters for set size.
                
                # A value is "lost" from the sumset if it was ONLY produced by pairs involving old_val
                # and is NOT produced by any pair involving new_val.
                # A value is "gained" to the sumset if it is produced by pairs involving new_val
                # and was NOT produced by any pair involving old_val.
                
                # More simply:
                # New Sumset = (Old Sumset - Old_Unique) | New_Unique
                # Where Old_Unique are sums that ONLY involved old_val.
                # This is hard to compute without counts.
                
                # Alternative efficient check:
                # Construct the "candidate" new sumset and prodset? No, too slow.
                
                # Let's use the property:
                # |S_new| = |S_old| - |S_old \ S_new| + |S_new \ S_old|
                # S_old \ S_new are sums that involved old_val and do NOT involve new_val.
                # S_new \ S_old are sums that involve new_val and did NOT involve old_val.
                
                # We can approximate or compute exactly.
                # Exact computation of set differences is O(K) where K is size of sets.
                # But we can just compute the size change directly if we track counts?
                # No, let's just compute the exact new size for the best candidates?
                # No, that's what we want to avoid.
                
                # Let's try a faster heuristic:
                # est_new_sum_size = len(sumset) + len(new_sum_elements - sumset) - len(old_sum_elements & (sumset - new_sum_elements))
                # This formula works if:
                # 1. new_sum_elements are the ONLY things added.
                # 2. old_sum_elements are the ONLY things removed.
                # 3. There is no overlap between "things added" and "things removed" that causes confusion.
                # Actually, if x is in old_sum_elements and in new_sum_elements, it stays.
                # If x is in old_sum_elements and NOT in new_sum_elements, it is removed IF it was ONLY in old.
                # If x is in new_sum_elements and NOT in old_sum_elements, it is added IF it was NOT in old.
                
                # The formula `len(new - old) - len(old & (old - new))` is not quite right.
                # Let's stick to a slightly more expensive but exact method for the top candidates,
                # or just use the set operations which are implemented in C and fast for small sets.
                
                # Given n=20, the number of sums is at most 210.
                # Creating sets of size ~20 and doing set operations is very fast.
                
                # Let's just compute the exact new size using set operations on the "delta" sets.
                # This is much faster than rebuilding the whole set.
                
                # Sums involving old_val:
                # S_old = {old_val + x for x in A} U {2*old_val}
                # S_new = {new_val + x for x in A} U {2*new_val}
                
                # The rest of the sumset (not involving A[i]) remains unchanged.
                # Let R be the sumset of A without A[i].
                # Current Sumset = R U S_old
                # New Sumset = R U S_new
                
                # We don't have R explicitly, but we know:
                # |Current| = |R U S_old|
                # |New| = |R U S_new|
                
                # This is still hard without R.
                
                # Let's go back to the incremental update of the FULL sets but optimized.
                # Since n=20, total pairs = 210.
                # Rebuilding the set takes ~210 additions.
                # Doing this for 1000 candidates is 210,000 operations.
                # In Python, 210k set additions is ~0.05s.
                # 1000 candidates * 20 indices = 20,000 evaluations per pass.
                # 20,000 * 0.05s = 1000s. Too slow.
                
                # We MUST use the delta approach.
                # Let's compute the exact change in size.
                
                # Elements that disappear from sumset:
                # Those in S_old that are NOT in R and NOT in S_new.
                # Elements that appear in sumset:
                # Those in S_new that are NOT in R and NOT in S_old.
                
                # We can find R by removing S_old from sumset? No, because S_old elements might be in R.
                
                # Okay, here is the robust fast method:
                # 1. Compute S_old and S_new (size <= 21).
                # 2. Identify candidates for removal: S_old.
                # 3. Identify candidates for addition: S_new.
                # 4. For each x in S_old:
                #    Check if x is in S_new. If yes, it stays (regardless of R).
                #    If no, check if x is in R. How to check R?
                #    We can precompute R? No, R changes if we consider different i.
                
                # Actually, we can just check if x is in the current sumset.
                # If x is in S_old and x is NOT in S_new:
                #    It is removed IF and ONLY IF it was NOT in R.
                #    We don't know if it was in R.
                
                # However, we can compute the new size exactly by:
                # new_sumset = (sumset - S_old) | S_new
                # This is O(|sumset| + |S_old| + |S_new|).
                # |sumset| is ~200-400.
                # This is ~500 operations per candidate.
                # 20,000 candidates * 500 ops = 10M ops.
                # 10M simple set ops in Python is ~1-2 seconds.
                # This is feasible within the time limit if we limit candidates or passes.
                
                # Let's try this.
                
                # Optimization: Only do this if the "potential" change looks good?
                # Hard to know without doing it.
                
                # Let's just do it. It's fast enough for 1000 candidates * 20 indices.
                
                new_sumset = (sumset - old_sum_elements) | new_sum_elements
                new_prodset = (prodset - old_prod_elements) | new_prod_elements
                
                est_score = max(len(new_sumset), len(new_prodset))
                
                if est_score < best_est_score:
                    best_est_score = est_score
                    best_new_val = new_val
                    best_new_sumset = new_sumset
                    best_new_prodset = new_prodset
            
            if best_new_val is not None and best_est_score < current_score:
                # Apply the change
                current_score = best_est_score
                A[i] = best_new_val
                sumset = best_new_sumset
                prodset = best_new_prodset
                improved = True
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
                
                # Break out of index loop to restart with new state?
                # Or continue? Continuing is better for "First Improvement" strategy.
                # But we updated sumset/prodset, so subsequent checks for other i
                # will use the updated sets. This is correct.
                
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds (limit is 90s)
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate the candidate pool once
    candidate_pool = generate_candidate_pool()
    
    best_A = None
    best_score = float('inf')
    
    # Generate diverse initial candidates
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    for a in range(1, 11):
        for d in range(1, 11):
            last = a + 19 * d
            if last > 10000:
                continue
            A = [a + i * d for i in range(20)]
            if len(set(A)) == 20:
                initial_candidates.append(A)
                
    # 2. Hybrid Sets: S U cS
    for k in range(5, 11):
        for c in range(k + 1, k + 20):
            S = list(range(1, k + 1))
            T = [c * x for x in S]
            A = S + T
            if len(set(A)) == 20 and max(A) <= 10000:
                initial_candidates.append(sorted(A))
                
    # 3. Geometric Progressions (small ratio)
    for a in range(1, 11):
        for r in range(2, 4):
            A = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                A.append(val)
                val *= r
            if len(A) == 20:
                initial_candidates.append(A)
                
    # 4. Random Smooth Sets
    smooth_nums = [x for x in candidate_pool if x <= 500]
    for _ in range(10):
        try:
            A = rng.sample(smooth_nums, 20)
            A = sorted(A)
            if max(A) <= 10000:
                initial_candidates.append(A)
        except ValueError:
            continue
            
    # Deduplicate initial candidates
    unique_candidates = []
    seen = set()
    for A in initial_candidates:
        key = tuple(sorted(A))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(A)
            
    # Sort initial candidates by their initial score to prioritize good starting points
    scored_candidates = []
    for A in unique_candidates:
        score = calc_score(A)
        scored_candidates.append((score, A))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Allocate time for local search on the best candidates
    num_candidates = len(scored_candidates)
    
    for idx, (init_score, A) in enumerate(scored_candidates):
        if time.time() > deadline - 2.0: # Keep 2s buffer
            break
            
        if init_score > best_score * 1.5 and idx > 5:
            continue
            
        remaining_time = deadline - time.time()
        # Allocate a fraction of the remaining time, but at least 0.5s, max 5s per candidate
        alloc_time = max(0.5, min(5.0, remaining_time / max(1, num_candidates - idx)))
        
        improved_A, score = local_search(A, alloc_time, candidate_pool, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
        
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
