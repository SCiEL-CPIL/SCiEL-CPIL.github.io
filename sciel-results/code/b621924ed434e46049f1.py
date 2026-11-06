"""Sum-free subset maximizer with greedy construction and local search."""

def construct():
    import time
    from random import Random
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(Sset, x):
        """Check if x can be added to S (represented as set) without violating sum-free property."""
        if x in Sset:
            return False
        # Check if 2*x is in Sset
        if 2 * x in Sset:
            return False
        # Check if x + a is in Sset for any a in Sset
        # This covers a + x = c
        for a in Sset:
            if a + x in Sset:
                return False
            # Check if a + b = x for any b in Sset
            # We can combine this loop or do separately. 
            # Let's do a+b=x check: for a in Sset, if x-a in Sset
            if x - a in Sset:
                return False
        return True

    def greedy_construct(S_full, rng):
        """Construct a sum-free subset by greedily adding elements."""
        S = []
        Sset = set()
        candidates = list(S_full)
        rng.shuffle(candidates)
        
        for x in candidates:
            if can_add(Sset, x):
                S.append(x)
                Sset.add(x)
        return S

    def local_search(S, S_full, rng, max_iters=5000):
        """Improve S by swapping elements."""
        S = list(S)
        Sset = set(S)
        S_full_set = set(S_full)
        removed = [x for x in S_full if x not in Sset]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not removed:
                break
            
            # Try to find a valid swap
            # We pick a random x from removed and try to find a y in S to remove
            # such that (S - {y}) U {x} is sum-free.
            
            # Optimization: Instead of random x, maybe pick x that has fewest conflicts?
            # For now, random x from removed.
            x = rng.choice(removed)
            
            found_swap = False
            # Shuffle S to randomize which y we try first? Or just iterate?
            # Iterating all y in S is O(N). Checking swap is O(N). Total O(N^2) per x.
            # N=240, N^2 ~ 50k ops. 5000 iters * 50k = 250M ops. Might be too slow in Python.
            # We need to be careful.
            
            # Let's limit the number of y's we try per x if it's too slow, or optimize the check.
            # Actually, can_swap check is O(N).
            # If we try all y in S for each x, it's O(N^2) per iteration of the outer loop.
            # 240^2 = 57600. 
            # 5000 iterations * 57600 = 288,000,000 operations.
            # Python can do ~10-50M simple ops/sec. This might take 5-30 seconds.
            # With 3 instances, it might be tight.
            
            # Strategy: Try a subset of y's or limit iterations.
            # Let's try to be smart: Only try to swap if x has a "chance".
            # But random is fine if we cap iterations.
            
            for y in S:
                # Check if swapping y for x is valid
                # S_new = (Sset - {y}) U {x}
                # Conditions:
                # 1. 2x not in Sset - {y}  => 2x in Sset and 2x != y
                # 2. No a in Sset - {y} such that a+x in Sset - {y}
                # 3. No a,b in Sset - {y} such that a+b=x
                
                valid = True
                
                # Check 1: 2x in Sset and 2x != y
                if 2 * x in Sset and 2 * x != y:
                    valid = False
                
                if valid:
                    # Check 2 and 3
                    # We iterate a in Sset. If a == y, skip.
                    for a in Sset:
                        if a == y:
                            continue
                        # Check a + x in Sset (and not y)
                        if a + x in Sset and a + x != y:
                            valid = False
                            break
                        # Check a + b = x => b = x - a in Sset (and not y)
                        if x - a in Sset and x - a != y:
                            valid = False
                            break
                    # Note: The above loop checks a+x in Sset and x-a in Sset.
                    # If a+x = y, it's removed, so ok.
                    # If x-a = y, it's removed, so ok.
                    
                if valid:
                    # Perform swap
                    Sset.remove(y)
                    S.remove(y)
                    Sset.add(x)
                    S.append(x)
                    removed.remove(x)
                    removed.append(y)
                    found_swap = True
                    break
            
            if not found_swap:
                # If no swap found for this x, maybe try another x?
                # To avoid getting stuck, we can just continue to next iteration 
                # which picks a new random x.
                pass

        return S

    def optimize_instance(size, seed):
        """Find large sum-free subset of the instance."""
        S_full = instance(size, seed)
        rng = Random(seed + 1000)
        
        best_S = []
        
        # Try a few greedy starts
        for i in range(5):
            S = greedy_construct(S_full, rng)
            if len(S) > len(best_S):
                best_S = S
        
        # Polish the best one with local search
        if best_S:
            best_S = local_search(best_S, S_full, rng, max_iters=2000)
            
        # Verify
        if not is_sum_free(best_S):
             # Fallback to simple odd numbers if something went wrong
             best_S = [x for x in S_full if x % 2]
             
        return best_S

    # Optimize each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback: use odd numbers as baseline
            S_full = instance(size, seed)
            results.append([x for x in S_full if x % 2])
            continue
        
        S_opt = optimize_instance(size, seed)
        
        # Verify sum-free
        if not is_sum_free(S_opt):
            # Fallback to greedy removal logic from previous version if needed, 
            # but greedy_add should always produce valid sets.
            # If invalid, fallback to odd numbers.
            S_full = instance(size, seed)
            S_opt = [x for x in S_full if x % 2]
        
        results.append(S_opt)
    
    return results
