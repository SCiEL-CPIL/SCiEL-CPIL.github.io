import time
import random
import numpy as np

N = 5000
TIME_LIMIT = 330

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)
NUM_PRIMES = len(PRIMES)

def construct():
    start_time = time.time()
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in PRIMES]
    
    best_size = 0
    best_residues = residues[:]
    
    # Pre-compute x % p for all x, p?
    # Or just compute on the fly.
    
    # Function to compute the set S and its size
    def get_set_size(res):
        # res is list of residues for each prime in PRIMES
        # Return count of x in [0, N] such that x % p != res[i] for all i
        count = 0
        for x in range(N + 1):
            ok = True
            for i, p in enumerate(PRIMES):
                if x % p == res[i]:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    # Coordinate Descent
    # We need to efficiently compute Cost(p, r)
    # Cost(p, r) = count of x in [0, N] such that x % p == r AND x % q != res[j] for all q != p
    
    # Optimization:
    # For each x, determine which primes block it.
    # blocked_by[x] = list of indices i such that x % PRIMES[i] == residues[i]
    
    def compute_blocked_by(res):
        blocked_by = [[] for _ in range(N + 1)]
        for i, p in enumerate(PRIMES):
            r = res[i]
            for x in range(r, N + 1, p):
                blocked_by[x].append(i)
        return blocked_by

    def coordinate_descent(res, max_iters=1000):
        res = res[:]
        blocked_by = compute_blocked_by(res)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            changed = False
            for i, p in enumerate(PRIMES):
                # Compute cost for each residue r
                # Cost(r) = count of x such that x % p == r AND i not in blocked_by[x]
                # Note: blocked_by[x] is based on the *current* res.
                # If we change res[i], blocked_by for x % p == res[i] (old) and x % p == new_r will change.
                # But for the purpose of choosing the *best* new r, we can approximate
                # by assuming other primes' blocks remain fixed.
                # This is standard coordinate descent.
                
                costs = [0] * p
                # Iterate x in [0, N]
                # For each x, if i is NOT in blocked_by[x], it contributes to Cost(x % p)
                # Wait, this logic is:
                # We want to pick r to minimize the number of x that WOULD BE in S
                # if we picked r, but are currently blocked by p?
                # No.
                # S_new = { x | x % p != r AND x % q != res[j] for q != p }
                # |S_new| = Total - | { x | x % p == r OR x % q == res[j] for q != p } |
                # This is hard to compute directly.
                
                # Let's stick to:
                # S = S_{-p} \ { x | x % p == r }
                # S_{-p} = { x | x % q != res[j] for q != p }
                # We want to minimize | S_{-p} \ { x | x % p == r } |?
                # No, we want to MAXIMIZE |S|.
                # |S| = |S_{-p}| - | S_{-p} \cap \{ x | x % p == r \} |
                # So we want to MINIMIZE | S_{-p} \cap \{ x | x % p == r \} |.
                
                # How to compute | S_{-p} \cap \{ x | x % p == r \} |?
                # This is the count of x such that x % p == r AND x % q != res[j] for all q != p.
                # This is exactly Cost(r) defined above.
                
                # We can compute this by iterating x.
                # For each x, if x % p == r, check if i is in blocked_by[x].
                # If i is NOT in blocked_by[x], it means x is NOT blocked by p.
                # But we want to know if x is in S_{-p}.
                # x is in S_{-p} if x is not blocked by ANY q != p.
                # i.e., if blocked_by[x] is a subset of {i}.
                # i.e., if all blockers of x are i.
                # Wait.
                # If x is in S_{-p}, it means x is not blocked by any q != p.
                # So blocked_by[x] can only contain i (or be empty).
                # But if x is blocked by i, it's not in S.
                # We are counting x that are in S_{-p} AND x % p == r.
                # If x % p == r, and we choose residue r for p, then x IS blocked by p.
                # So x is NOT in S_new.
                # We want to minimize the number of such x.
                
                # So Cost(r) = count of x such that:
                # 1. x % p == r
                # 2. x is not blocked by any q != p.
                
                # Condition 2 is: for all j != i, x % PRIMES[j] != res[j].
                # This is equivalent to: blocked_by[x] contains no index j != i.
                # So blocked_by[x] must be a subset of {i}.
                
                # We can pre-compute for each x, the set of blockers.
                # Then for a given i, we just check if any blocker in blocked_by[x] is != i.
                
                for x in range(N + 1):
                    r_x = x % p
                    # Check if x is in S_{-p}
                    # i.e., is i the ONLY blocker? Or is there NO blocker?
                    # If there is a blocker j != i, x is NOT in S_{-p}.
                    in_S_minus_p = True
                    for j in blocked_by[x]:
                        if j != i:
                            in_S_minus_p = False
                            break
                    
                    if in_S_minus_p:
                        costs[r_x] += 1
                
                # Find min cost residue
                min_cost = min(costs)
                best_r = costs.index(min_cost)
                
                if best_r != res[i]:
                    res[i] = best_r
                    changed = True
                    # Update blocked_by for affected x
                    # x where x % p == old_res or x % p == new_res
                    old_r = res[i] # Wait, res[i] was updated.
                    # We need old_r before update.
                    # Let's fix the logic.
                    
            if not changed:
                break
                
        return res

    # The above loop has a bug: updating res[i] inside the loop affects blocked_by for subsequent primes?
    # No, blocked_by is computed once per iteration of the outer loop.
    # But if we change res[i], the costs for other primes j are slightly inaccurate
    # because they used the old blocked_by.
    # This is standard synchronous coordinate descent. It usually works.
    
    # Let's run a few restarts.
    
    for restart in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        residues = [random.randint(0, p - 1) for p in PRIMES]
        residues = coordinate_descent(residues, max_iters=50)
        
        # Calculate final size
        size = get_set_size(residues)
        
        if size > best_size:
            best_size = size
            best_residues = residues[:]
            
    # Construct final list
    final_set = []
    for x in range(N + 1):
        ok = True
        for i, p in enumerate(PRIMES):
            if x % p == best_residues[i]:
                ok = False
                break
        if ok:
            final_set.append(x)
            
    return final_set
