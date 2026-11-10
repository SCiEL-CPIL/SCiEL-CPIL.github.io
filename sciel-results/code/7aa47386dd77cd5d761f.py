import time
import math
import random

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # 1. Generate a diverse pool of candidate numbers
    # Focus on smooth numbers and small integers
    candidates = set()
    
    # Small integers
    for i in range(1, 151):
        candidates.add(i)
        
    # 2-smooth (powers of 2)
    for a in range(0, 14):
        candidates.add(2**a)
        
    # 3-smooth
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2**a) * (3**b)
            if val <= 10000:
                candidates.add(val)
                
    # 5-smooth
    for a in range(0, 14):
        for b in range(0, 14):
            for c in range(0, 14):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    candidates.add(val)
                    
    # 7-smooth (Humble numbers) - add a few more
    for a in range(0, 10):
        for b in range(0, 10):
            for c in range(0, 10):
                for d in range(0, 10):
                    val = (2**a) * (3**b) * (5**c) * (7**d)
                    if val <= 10000:
                        candidates.add(val)
                        
    # Add some randoms to break symmetry if needed
    for _ in range(200):
        candidates.add(random.randint(1, 10000))
        
    candidates = sorted(list(candidates))
    
    # 2. Greedy Construction
    # Start with a small seed that is known to be good: {1, 2, 3, 4, 5}
    # Actually, starting with {1, 2} is a good base.
    # Let's start with {1, 2, 3} and expand.
    
    A = [1, 2, 3]
    
    # Precompute current sets for incremental update
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    current_score = max(len(sumset), len(prodset))
    
    # We will greedily add elements until we reach 20
    while len(A) < 20:
        if time.time() > deadline:
            break
            
        best_c = None
        best_new_score = float('inf')
        
        # To speed up, we can limit the search to a subset of candidates
        # But let's try all candidates first. If it's too slow, we'll restrict.
        # 20 elements, ~1000 candidates. 20*1000 = 20000 checks per iteration.
        # Each check is O(N) for sums/prods. N=20. So 400k ops per iteration.
        # 17 iterations left. ~7M ops. Should be fast enough in Python?
        # Python is slow. Let's optimize the check.
        
        # Optimization: 
        # For a candidate c, new sums are {a+c for a in A} U {c+c}
        # new prods are {a*c for a in A} U {c*c}
        
        for c in candidates:
            if c in A:
                continue
                
            # Calculate marginal increase
            # New sums added:
            new_sums = set()
            for a in A:
                s = a + c
                if s not in sumset:
                    new_sums.add(s)
            c_c = c + c
            if c_c not in sumset:
                new_sums.add(c_c)
                
            # New prods added:
            new_prods = set()
            for a in A:
                p = a * c
                if p not in prodset:
                    new_prods.add(p)
            c_c_p = c * c
            if c_c_p not in prodset:
                new_prods.add(c_c_p)
                
            new_sum_size = len(sumset) + len(new_sums)
            new_prod_size = len(prodset) + len(new_prods)
            new_score = max(new_sum_size, new_prod_size)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_c = c
                best_new_sums = new_sums
                best_new_prods = new_prods
                
        if best_c is None:
            # Fallback: pick smallest available
            for c in candidates:
                if c not in A:
                    best_c = c
                    break
                    
        if best_c is None:
            break
            
        # Update state
        A.append(best_c)
        sumset.update(best_new_sums)
        prodset.update(best_new_prods)
        current_score = best_new_score
        
    # 3. Local Search (Hill Climbing)
    # Try to swap any element in A with any element in candidates
    # We will do multiple passes until no improvement is found or time runs out.
    
    # To make local search efficient, we need to quickly calculate the score change
    # when removing x and adding y.
    # Removing x:
    #   Sums to remove: {x+a for a in A if a!=x} U {2x}
    #   Prods to remove: {x*a for a in A if a!=x} U {x*x}
    # Adding y:
    #   Sums to add: {y+a for a in A_new} U {2y}
    #   Prods to add: {y*a for a in A_new} U {y*y}
    # Note: A_new = A \ {x} U {y}
    
    # This is complex to do incrementally without rebuilding sets.
    # Given N=20, rebuilding sets is O(400) operations.
    # If we test 1000 candidates for each of 20 elements, that's 20,000 rebuilds.
    # 20,000 * 400 = 8,000,000 ops. This is feasible in Python if optimized.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_swap_score = current_score
        
        for i, x in enumerate(A):
            if time.time() > deadline:
                break
                
            # Try swapping x with every candidate y
            for y in candidates:
                if y == x or y in A:
                    continue
                
                # Construct new set
                A_new = A[:i] + A[i+1:] + [y]
                
                # Calculate score
                # Optimization: We can just call calc_score(A_new)
                # But let's try to be a bit smarter.
                # For now, calc_score is O(N^2). N=20 -> 400 ops.
                # 20 * 1000 candidates = 20,000 calls.
                # 20,000 * 400 = 8M ops.
                # This might be tight in 60s if Python is slow.
                # Let's limit the candidate pool for local search to the most promising ones.
                # The most promising are those close to existing values or smooth numbers.
                # Let's just use the full candidate list but break early if time is tight.
                
                s = calc_score(A_new)
                
                if s < best_swap_score:
                    best_swap_score = s
                    best_swap = (i, x, y)
                    improved = True
        
        if improved:
            i, x, y = best_swap
            A[i] = y
            current_score = best_swap_score
        else:
            break
            
    # 4. Final Polish
    # If we have time left, do another pass with a smaller, more targeted candidate pool
    # e.g., only numbers within +/- 10% of existing numbers
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_swap = None
            best_swap_score = current_score
            
            # Generate targeted candidates
            targeted = set()
            for x in A:
                for delta in range(-10, 11):
                    if 1 <= x + delta <= 10000:
                        targeted.add(x + delta)
                # Also add multiples/factors
                if x > 1:
                    targeted.add(x * 2)
                    if x % 2 == 0:
                        targeted.add(x // 2)
            
            targeted = sorted(list(targeted))
            
            for i, x in enumerate(A):
                for y in targeted:
                    if y == x or y in A:
                        continue
                    
                    A_new = A[:i] + A[i+1:] + [y]
                    s = calc_score(A_new)
                    
                    if s < best_swap_score:
                        best_swap_score = s
                        best_swap = (i, x, y)
                        improved = True
            
            if improved:
                i, x, y = best_swap
                A[i] = y
                current_score = best_swap_score
            else:
                break

    # Ensure valid output
    best_set = sorted(list(set(A)))
    if len(best_set) < 20:
        for i in range(1, 100):
            if i not in best_set:
                best_set.append(i)
            if len(best_set) == 20:
                break
        best_set = best_set[:20]
    elif len(best_set) > 20:
        best_set = best_set[:20]
        
    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = calc_score(res)
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {score}")
    print(f"Set: {res}")
