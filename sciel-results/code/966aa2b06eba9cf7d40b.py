import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 120  # Safety margin for 150s limit
    
    # Sieve for primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    
    # State: list of (p, r)
    # We will maintain a set of primes used and their residues
    # Sum of reciprocals constraint: sum(1/p) <= 1
    
    # Helper: Calculate max contiguous L covered by a set of (p, r)
    # L is capped at 5000 for evaluation speed, but we expect L < 1000
    def calc_L(classes, max_L=2000):
        covered = [False] * (max_L + 1)
        for p, r in classes:
            for x in range(r, max_L + 1, p):
                if x > 0:
                    covered[x] = True
                # Note: if r==0, it covers p, 2p, ...
                # If r>0, it covers r, r+p, ...
                # We need to cover 1..L.
                # If r==0, 0 is covered, but we care about 1..L.
                # The loop starts at r. If r==0, first is 0. Next is p.
                # We need 1..L.
                # So if r==0, 1..L are covered if p <= L? No.
                # 1 is covered if 1 == r mod p. If r=0, 1 is not covered by this class.
                # So we just mark the indices.
        
        # Find largest L such that 1..L are all True
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Helper: Check if sum of reciprocals <= 1
    def sum_recip(classes):
        # Use fractions for exactness? 
        # sum of 1/p for p<=1000 is small. Float is likely sufficient for comparison with 1.0
        # but "exact rationals" is requested in task description.
        # However, for search, float is much faster. 
        # We can verify exactness at the end.
        s = 0.0
        for p, r in classes:
            s += 1.0 / p
        return s

    # Greedy initialization
    # Start with empty set
    classes = []
    current_sum = 0.0
    
    # We want to cover 1..L. 
    # Greedy step: Find the smallest uncovered integer u.
    # Pick (p, r) that covers u and maximizes "value".
    
    # To speed up, we'll just try to build a set of primes first, then optimize residues.
    # Which primes to pick?
    # Small primes are generally better for covering small numbers.
    # Let's try a greedy set selection.
    
    # Candidate primes: all primes <= 1000
    # We will iteratively add a prime (with best residue) if it increases L.
    
    # Initial set: Try adding primes in increasing order? 
    # Or random?
    # Let's try a greedy approach on residues directly.
    
    # Start with a few small primes to get a base
    # 2, 3, 5, 7, 11, 13...
    # Sum 1/2+1/3+1/5+1/7+1/11+1/13 = 0.5+0.33+0.2+0.14+0.09+0.07 = 1.33 > 1.
    # So we can't take all small ones.
    
    # Let's use a local search on the set of primes.
    # Initialize with a random subset or greedy subset.
    
    # Greedy subset selection:
    # Start with empty.
    # While sum < 1:
    #   For each prime p not in set:
    #     For each residue r:
    #       Calculate L if (p, r) is added.
    #     Pick best (p, r) that increases L most.
    #     Add it.
    
    # This is O(P * P * L) which is too slow if done naively.
    # P ~ 168. L ~ 100. 168*168*100 = 2.8M ops per step? 
    # Number of steps ~ 20. Total 50M ops. Might be tight in Python.
    
    # Optimization: 
    # We can pre-compute coverage for each (p, r) as a bitmask or boolean list.
    # Then adding a class is just OR-ing the masks.
    
    max_L_eval = 500
    # Pre-compute masks for all (p, r)
    # masks[p][r] = list of bools or int bitmask?
    # 500 bits fits in an int.
    
    masks = {}
    for p in primes:
        masks[p] = [0] * p
        for r in range(p):
            m = 0
            for x in range(r, max_L_eval + 1, p):
                if x > 0:
                    m |= (1 << x)
            masks[p][r] = m
    
    # Greedy construction
    used_primes = set()
    current_mask = 0
    current_sum = 0.0
    classes = []
    
    def get_L_from_mask(m):
        # Find highest bit set? No, contiguous from 1.
        # Check bits 1, 2, ...
        L = 0
        for i in range(1, max_L_eval + 1):
            if (m >> i) & 1:
                L = i
            else:
                break
        return L
    
    best_L = 0
    best_classes = []
    
    while time.time() - start_time < time_limit - 5:
        if current_sum >= 1.0:
            break
            
        best_gain = -1
        best_p = None
        best_r = None
        
        for p in primes:
            if p in used_primes:
                continue
            for r in range(p):
                new_mask = current_mask | masks[p][r]
                new_L = get_L_from_mask(new_mask)
                gain = new_L - best_L
                # We want to maximize L. 
                # To be efficient, we might prioritize primes that cover the current gap.
                # Simple gain is fine.
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
        
        if best_p is None or best_gain <= 0:
            # No improvement found by adding a single prime
            break
            
        # Add the best prime
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        current_sum += 1.0 / best_p
        current_mask |= masks[best_p][best_r]
        best_L = get_L_from_mask(current_mask)
        best_classes = [c[:] for c in classes]
        
    # Now we have a set of primes. 
    # Optimize residues for this fixed set of primes using local search.
    
    def optimize_residues(cls):
        # cls is list of [p, r]
        # Returns best L and best residues
        current_L = get_L_from_mask(0) # Recalculate? 
        # Let's just calculate L from scratch for current residues
        m = 0
        for p, r in cls:
            m |= masks[p][r]
        current_L = get_L_from_mask(m)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            for i, (p, r) in enumerate(cls):
                best_r_for_p = r
                best_L_for_p = current_L
                for r_try in range(p):
                    if r_try == r:
                        continue
                    # Swap r with r_try
                    # m_new = m_without_p_r | mask_p_r_try
                    # This is tricky with bitmasks if we don't store individual masks.
                    # Easier: recalculate mask for all classes. 
                    # K is small (~10-20). L is 500.
                    # Recalculating mask: 20 * (500/p) ops. Very fast.
                    
                    m_new = 0
                    for j, (pj, rj) in enumerate(cls):
                        if j == i:
                            m_new |= masks[pj][r_try]
                        else:
                            m_new |= masks[pj][rj]
                    
                    L_new = get_L_from_mask(m_new)
                    if L_new > best_L_for_p:
                        best_L_for_p = L_new
                        best_r_for_p = r_try
                
                if best_r_for_p != r:
                    cls[i][1] = best_r_for_p
                    current_L = best_L_for_p
                    improved = True
                    # Update mask? 
                    # We'll just recalculate mask at end of loop if needed, 
                    # or just rely on next iteration's recalculation.
                    # To save time, we can update current mask here.
                    # But we don't have the old mask of p,r stored separately easily.
                    # Just recalculate full mask.
                    m = 0
                    for pj, rj in cls:
                        m |= masks[pj][rj]
                    
        return current_L, cls

    # Run local search on the greedy set
    L_opt, cls_opt = optimize_residues(best_classes)
    
    # Try to improve the set of primes
    # Swap primes
    # We can try to replace a prime p_i with a new prime p_j
    # If 1/p_j <= 1/p_i (to keep sum down) or if we have room.
    
    # Let's do a simple hill climbing on the set of primes.
    # State: list of primes.
    # Objective: Max L after residue optimization.
    
    # To avoid re-running full residue optimization from scratch every time,
    # we can keep the residues and just re-optimize.
    
    current_set = [p for p, r in best_classes]
    current_res = [r for p, r in best_classes]
    current_L = L_opt
    
    # Pre-compute sum for current set
    curr_sum = sum(1.0/p for p in current_set)
    
    # List of candidate primes to swap in
    candidates = [p for p in primes if p not in current_set]
    
    # Shuffle candidates for diversity? 
    # Just iterate.
    
    for _ in range(100): # Max swaps
        if time.time() - start_time > time_limit - 2:
            break
            
        improved = False
        # Try swapping each current prime with each candidate
        # This is O(N^2). N~20, Cand~150. 3000 swaps.
        # Each swap requires residue optimization.
        # Residue optimization takes ~ K * P * L. 
        # Might be too slow.
        
        # Alternative: 
        # 1. Remove one prime. Optimize residues. Record L.
        # 2. Add one prime. Optimize residues. Record L.
        
        # Let's try adding a prime if sum allows.
        if curr_sum < 1.0:
            best_add_p = None
            best_add_L = current_L
            for p in candidates:
                if curr_sum + 1.0/p > 1.0:
                    continue
                # Try adding p with best residue
                # We need to find best r for p given current residues.
                # We can just call optimize_residues on the new set.
                # But that's expensive.
                
                # Heuristic: 
                # Just try a few residues for p?
                # Or just add p with r=0 and let optimize_residues fix it.
                
                # Let's just pick the smallest candidate prime?
                # Or random?
                
                # Let's try the smallest unused prime.
                break # Just take the first valid one for now to test speed
            
            if best_add_p:
                new_set = current_set + [best_add_p]
                new_res = current_res + [0]
                new_sum = curr_sum + 1.0/best_add_p
                L_new, res_new = optimize_residues([[p, r] for p, r in zip(new_set, new_res)])
                if L_new > current_L:
                    current_set = new_set
                    current_res = res_new
                    current_L = L_new
                    curr_sum = new_sum
                    candidates.remove(best_add_p)
                    improved = True
        
        if not improved:
            # Try swapping
            # Pick a random prime to remove?
            # Or the one with largest 1/p (smallest prime) as it's "costly"?
            # Removing a small prime frees up budget.
            
            # Let's try removing the smallest prime in the set
            if current_set:
                idx_min = 0
                min_p = current_set[0]
                for i, p in enumerate(current_set):
                    if p < min_p:
                        min_p = p
                        idx_min = i
                
                # Remove it
                test_set = current_set[:idx_min] + current_set[idx_min+1:]
                test_res = current_res[:idx_min] + current_res[idx_min+1:]
                
                # Try adding a new prime
                # We have freed up 1/min_p.
                # We can add primes with sum <= 1/min_p.
                
                # Just try adding the smallest available prime
                # Or a random one.
                
                # For simplicity, just keep the smaller set if it doesn't hurt L too much?
                # No, we want to maximize L.
                
                # Let's just stop set optimization for now. 
                # Residue optimization is the main driver.
                break

    # Final verification with exact rationals
    from fractions import Fraction
    total_sum = Fraction(0, 1)
    for p, r in best_classes:
        total_sum += Fraction(1, p)
    
    if total_sum > 1:
        # Should not happen if we tracked float correctly, but just in case
        # Remove primes until sum <= 1
        # Sort by 1/p descending (smallest primes first)
        best_classes.sort(key=lambda x: x[0])
        while total_sum > 1 and best_classes:
            p, r = best_classes.pop(0)
            total_sum -= Fraction(1, p)
            # Re-optimize residues? 
            # If we removed a prime, L might drop.
            # We should re-optimize.
            L_opt, cls_opt = optimize_residues(best_classes)
            best_classes = cls_opt
            # Recalc sum
            total_sum = Fraction(0, 1)
            for p, r in best_classes:
                total_sum += Fraction(1, p)

    # Final L calculation
    final_L = calc_L(best_classes, max_L=10000)
    
    return best_classes

# EVOLVE-BLOCK-END
