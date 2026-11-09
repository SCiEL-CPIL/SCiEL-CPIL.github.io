import time
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    # Constructive Greedy with Lookahead
    # State: Set S represented as a boolean array
    # We maintain:
    # 1. in_set: bool array of size N+1
    # 2. current_len: |S|
    # 3. missed_residues[p]: set of residues mod p not in S (for primes p <= current_len)
    #    Actually, we only care if missed_residues[p] is empty.
    #    To check if adding x is valid, we need to know if x mod p is in missed_residues[p].
    #    If x mod p is in missed_residues[p] AND missed_residues[p] has size 1, then adding x makes missed_residues[p] empty -> Invalid.
    #    If missed_residues[p] has size > 1, adding x is safe for p (it just reduces the count).
    
    in_set = [False] * (N + 1)
    current_len = 0
    
    primes = _primes(N + 1)
    # We only need to track primes up to N+1, but the constraint is p <= current_len.
    # As current_len grows, we might need to track more primes.
    # Max current_len is N+1.
    
    # missed_residues[p] will be a set of ints.
    # To make it fast, we can store the count of missed residues and a list/set of which ones are missed.
    # Since p is small (<= 5001), we can use a bitmask or a small list.
    # Let's use a list of booleans or a set.
    
    # Initialize
    # For any p, initially all residues are missed.
    # missed_count[p] = p
    # missed_set[p] = set(range(p))
    
    # We don't need to initialize all primes at once, just update as we go.
    # But checking validity requires knowing the status of p <= current_len.
    
    # Let's pre-compute prime indices for quick lookup.
    # prime_to_index = {p: i for i, p in enumerate(primes)}
    
    # Efficient data structures:
    # missed_count[p]: int
    # missed_list[p]: list of residues currently missed.
    # When we add x, for each p <= current_len:
    #   r = x % p
    #   if r in missed_list[p]:
    #       missed_count[p] -= 1
    #       # remove r from missed_list[p]
    #       # if missed_count[p] == 0, violation occurred (but we check before adding)
    
    # To check if x is valid:
    # 1. For all p <= current_len:
    #      r = x % p
    #      if r in missed_set[p] and missed_count[p] == 1: return False
    # 2. For p = current_len + 1 (if prime):
    #      Let p_new = current_len + 1
    #      We need to ensure S U {x} misses at least one residue mod p_new.
    #      Currently S covers some residues. x covers r_new = x % p_new.
    #      Let covered = set(y % p_new for y in S)
    #      We need (covered U {r_new}) != set(range(p_new))
    #      This is equivalent to: there exists a residue r in range(p_new) such that r not in covered and r != r_new.
    #      Or: missed_count_new = p_new - len(covered U {r_new}) > 0.
    #      Note: We don't have "covered" stored directly, we have "missed".
    #      missed_count_new = p_new - (len(covered) + (1 if r_new not in covered else 0))
    #      We can compute len(covered) as p_new - missed_count_initial_for_p_new?
    #      No, missed_count is dynamic.
    #      Let's just compute the residues covered by S mod p_new on the fly if p_new is prime.
    #      Since p_new is small (<= 5001), and |S| is up to 5000, this is O(N) per candidate.
    #      With N=5000 candidates, this is 25M ops per step. Too slow if done naively for every x.
    
    # Optimization for Step 2:
    # Precompute for each prime p, the residues covered by S.
    # Actually, we can maintain `covered_mask[p]` as a bitmask or boolean array.
    # But updating it for every x added is O(N) if we iterate S.
    # Instead, when we add x, we update the state for all p.
    # For p <= current_len, we update missed info.
    # For p_new = current_len + 1, we need to know if S U {x} is admissible.
    # We can maintain `covered_count[p]` and `covered_set[p]` for all primes p?
    # Memory: 5000 primes * 5000 bits? No, p is up to 5000.
    # We can maintain `residue_covered[p][r]` boolean array?
    # Total memory sum(p for p in primes) ~ N^2 / log N ~ 25e6 / 8 ~ 3MB. Feasible.
    # But updating it: when we add x, we set residue_covered[p][x%p] = True for all p.
    # This is O(pi(N)) per addition.
    # pi(5000) ~ 669.
    # 5000 additions * 669 updates ~ 3.3M updates. Very fast.
    
    # So, we maintain:
    # covered[p]: boolean array of size p.
    # missed_count[p]: int.
    # 
    # Initialization:
    # covered[p] = [False] * p
    # missed_count[p] = p
    
    # Add x:
    # 1. Check validity:
    #    a. For p <= current_len:
    #       r = x % p
    #       if not covered[p][r]:
    #           if missed_count[p] == 1: return False (would become 0)
    #    b. p_new = current_len + 1
    #       if p_new is prime:
    #          r_new = x % p_new
    #          # Check if S U {x} covers all residues mod p_new
    #          # Current covered count: sum(covered[p_new])
    #          # If r_new is not covered, new_count = cur_count + 1
    #          # If r_new is covered, new_count = cur_count
    #          # If new_count == p_new, return False
    #          cur_count = sum(covered[p_new]) # This is O(p_new). 
    #          # To avoid O(p) sum, maintain covered_count[p].
    #          if covered_count[p_new] + (0 if covered[p_new][r_new] else 1) == p_new:
    #              return False
    #
    # 2. Commit x:
    #    in_set[x] = True
    #    current_len += 1
    #    for p in primes:
    #       if p > current_len: break # Only need to update p <= current_len? 
    #       # Actually, for future checks, we need covered[p] to be correct for p <= current_len.
    #       # When current_len increases, the new prime p_new = current_len (if prime) needs its covered array to be correct.
    #       # But we haven't been updating covered[p_new] for previous elements if p_new > old_current_len.
    #       # So we MUST update covered[p] for ALL p <= N+1 whenever we add an element?
    #       # That would be O(pi(N)) per addition, which is fine.
    #       # But wait, if we update covered[p] for p > current_len, we are doing extra work.
    #       # However, we need covered[p_new] to be correct when p_new becomes <= current_len.
    #       # So yes, update all p.
            
            # r = x % p
            # if not covered[p][r]:
            #     covered[p][r] = True
            #     covered_count[p] += 1
            #     missed_count[p] -= 1

    # This seems robust.
    
    # Let's refine the "Check Validity" for p_new.
    # We need covered_count[p_new] and covered[p_new][r_new].
    
    # Implementation details:
    # primes list
    # is_prime array for quick lookup
    # covered: dict or list of lists? List of lists indexed by prime value? 
    # Primes are up to 5001. We can use an array `covered_arr` of size N+2, where covered_arr[p] is a byte array of size p if p is prime, else None.
    # covered_count_arr of size N+2.
    # missed_count_arr of size N+2.
    
    # Pre-computation
    is_prime = [False] * (N + 2)
    for p in primes:
        is_prime[p] = True
        
    # Initialize covered structures
    # Only for primes
    covered = {}
    covered_count = [0] * (N + 2)
    missed_count = [0] * (N + 2)
    for p in primes:
        covered[p] = bytearray(p) # 0 or 1
        covered_count[p] = 0
        missed_count[p] = p
        
    # Main Loop
    while time.time() < _DEADLINE:
        # Find best x to add
        best_x = -1
        best_score = -1
        
        # Heuristic: Try all x? N=5000.
        # Check cost: O(pi(N)) ~ 700 ops per x.
        # Total per iteration: 5000 * 700 = 3.5M ops.
        # 300s / 3.5M ~ 85000 iterations? No, 3.5M ops is very fast in Python? 
        # Python loop overhead is high. 3.5M iterations in Python might take 1-2 seconds.
        # So ~150 iterations in 300s.
        # This might be too slow if we do a full scan every time.
        
        # Optimization: Random sampling?
        # Or, maintain a list of candidates.
        
        # Let's try a simpler heuristic first: 
        # Just pick the first valid x?
        # Or pick x that minimizes "future regret".
        
        # Given the time limit, a full scan of 5000 candidates with 700 checks each might be borderline.
        # 5000 * 700 = 3.5e6.
        # In Python, a simple loop takes ~50ns. 3.5e6 * 50ns = 0.17s.
        # Plus overhead. Maybe 0.5s per iteration.
        # 300s / 0.5s = 600 iterations.
        # 600 elements added. Size 600.
        # The greedy removal got 0.77 * Seed. Seed is likely large?
        # If Seed is 1000, we want > 770.
        # 600 might be too small.
        
        # We need a faster validity check or a better selection.
        
        # Alternative: 
        # The constraint for p_new is the bottleneck for calculation?
        # No, the loop over p <= current_len is the main cost.
        
        # Can we prune?
        # If x is invalid for some small p, it's invalid.
        # Small p are more likely to be violated?
        
        # Let's try to optimize the inner loop using NumPy or C-like efficiency?
        # We can precompute x % p for all x?
        # x_mod_p[x][p]? Memory 5000*5000 too big.
        
        # Let's stick to the constructive greedy but optimize the scan.
        # We can iterate through x from 0 to N.
        
        # To speed up, we can check the most restrictive primes first.
        # Small primes are more restrictive?
        
        # Let's implement and see.
        
        candidates = []
        for x in range(N + 1):
            if in_set[x]:
                continue
            
            valid = True
            # Check p <= current_len
            for p in primes:
                if p > current_len:
                    break
                r = x % p
                if not covered[p][r]:
                    if missed_count[p] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check p_new
            p_new = current_len + 1
            if is_prime[p_new]:
                r_new = x % p_new
                if covered_count[p_new] + (0 if covered[p_new][r_new] else 1) == p_new:
                    valid = False
            
            if valid:
                # Score: ?
                # Prefer x that keeps more options open?
                # Simple: just take it.
                # Or, score based on how many "tight" constraints it relaxes?
                # Tight constraint: missed_count[p] == 1.
                # If x covers a residue in a tight p, it's bad (but we already checked it's not the ONLY one).
                # Actually, if missed_count[p] > 1, and x covers a missed residue, missed_count decreases.
                # We want to keep missed_count high? No, we just need it > 0.
                # So we want to avoid reducing missed_count[p] to 1 if it was 2?
                # Score: sum(1 for p if missed_count[p]==1 and x covers missed residue) -> Minimize this.
                
                # For now, just pick the first valid one or random.
                # Let's pick the one with the smallest x? No, that's biased.
                # Let's pick randomly from a small sample?
                # Or just the first one found.
                
                best_x = x
                break # Take the first valid one to save time?
                # Or collect a few and pick best?
        
        if best_x == -1:
            break
        
        # Commit best_x
        in_set[best_x] = True
        current_len += 1
        
        # Update state
        for p in primes:
            if p > N + 1:
                break
            # We only need to update p <= current_len for the "missed" logic of future steps?
            # But for p_new checks, we need covered[p_new] to be correct.
            # So we must update all p.
            
            r = best_x % p
            if not covered[p][r]:
                covered[p][r] = 1
                covered_count[p] += 1
                missed_count[p] -= 1

    result = [v for v in range(N + 1) if in_set[v]]
    return result
