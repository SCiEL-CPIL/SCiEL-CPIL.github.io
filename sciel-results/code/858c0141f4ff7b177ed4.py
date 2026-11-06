import time
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return list(np.where(sieve)[0])

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285
    primes = _primes(N + 1)
    
    # Precompute residue class memberships as bitmasks for fast intersection
    # mask[p][r] has bit i set if i % p == r
    masks = {}
    for p in primes:
        if p > N + 1:
            break
        classes = [0] * p
        for x in range(N + 1):
            r = x % p
            classes[r] |= (1 << x)
        masks[p] = classes
    
    # For each prime p, precompute the size of each residue class
    sizes = {}
    for p in primes:
        if p > N + 1:
            break
        sizes[p] = [bin(masks[p][r]).count('1') for r in range(p)]
    
    best_S = None
    best_k = 0
    
    # Strategy 1: Greedy removal (complement E) - batch removal
    def greedy_remove():
        E_mask = 0
        # Track counts of E in each residue class for each prime
        # count[p][r] = number of elements in E that are in class r mod p
        count = {}
        for p in primes:
            if p > N + 1:
                break
            count[p] = [0] * p
        
        E_size = 0
        k = N + 1
        
        while time.time() < deadline - 10:
            active_primes = [p for p in primes if p <= k]
            
            unsatisfied = []
            for p in active_primes:
                # Check if any residue class is fully covered
                satisfied = False
                for r in range(p):
                    if count[p][r] == sizes[p][r]:
                        satisfied = True
                        break
                if not satisfied:
                    # Find best residue class to add (minimizes new elements)
                    min_new = float('inf')
                    best_r = 0
                    for r in range(p):
                        new = sizes[p][r] - count[p][r]
                        if new < min_new:
                            min_new = new
                            best_r = r
                    unsatisfied.append((p, min_new, best_r))
            
            if not unsatisfied:
                break
            
            # Pick the prime with smallest p (most restrictive)
            unsatisfied.sort(key=lambda x: x[0])
            p_sel, cost, r_sel = unsatisfied[0]
            
            # Add all elements in class r_sel mod p_sel to E
            class_elements = [x for x in range(N + 1) if x % p_sel == r_sel]
            for x in class_elements:
                if not (E_mask >> x) & 1:
                    E_mask |= (1 << x)
                    E_size += 1
                    # Update counts for all primes
                    for q in primes:
                        if q > N + 1:
                            break
                        count[q][x % q] += 1
            
            k = N + 1 - E_size
            if k <= 0:
                break
        
        S = [x for x in range(N + 1) if not (E_mask >> x) & 1]
        return S
    
    S1 = greedy_remove()
    if S1:
        k1 = len(S1)
        best_k = max(best_k, k1)
        best_S = S1
    
    # Strategy 2: Direct construction - start with all elements, remove minimally
    # This is essentially the same as Strategy 1 but let's try a different tie-breaking
    def greedy_remove_lowest_cost():
        E_mask = 0
        count = {}
        for p in primes:
            if p > N + 1:
                break
            count[p] = [0] * p
        
        E_size = 0
        k = N + 1
        
        while time.time() < deadline - 5:
            active_primes = [p for p in primes if p <= k]
            
            unsatisfied = []
            for p in active_primes:
                satisfied = False
                for r in range(p):
                    if count[p][r] == sizes[p][r]:
                        satisfied = True
                        break
                if not satisfied:
                    min_new = float('inf')
                    best_r = 0
                    for r in range(p):
                        new = sizes[p][r] - count[p][r]
                        if new < min_new:
                            min_new = new
                            best_r = r
                    unsatisfied.append((p, min_new, best_r))
            
            if not unsatisfied:
                break
            
            # Pick lowest cost
            unsatisfied.sort(key=lambda x: x[1])
            p_sel, cost, r_sel = unsatisfied[0]
            
            class_elements = [x for x in range(N + 1) if x % p_sel == r_sel]
            for x in class_elements:
                if not (E_mask >> x) & 1:
                    E_mask |= (1 << x)
                    E_size += 1
                    for q in primes:
                        if q > N + 1:
                            break
                        count[q][x % q] += 1
            
            k = N + 1 - E_size
            if k <= 0:
                break
        
        S = [x for x in range(N + 1) if not (E_mask >> x) & 1]
        return S
    
    S2 = greedy_remove_lowest_cost()
    if S2 and len(S2) > best_k:
        best_k = len(S2)
        best_S = S2
    
    # Strategy 3: Conflict graph heuristic
    # Build a simplified conflict structure and use greedy independent set
    def conflict_greedy():
        # Start with all elements
        S_set = set(range(N + 1))
        k = N + 1
        primes_list = [p for p in primes if p <= N + 1]
        
        # For each prime p <= k, we need to ensure not all residues are present
        # Conflict: for each prime p, if all residues are present, we must remove one class
        
        while time.time() < deadline - 2:
            k = len(S_set)
            active_primes = [p for p in primes_list if p <= k]
            
            violations = []
            for p in active_primes:
                residues_present = set(x % p for x in S_set)
                if len(residues_present) == p:
                    # Need to remove one residue class
                    # Find the class with fewest elements (cheapest removal)
                    class_counts = {}
                    for r in range(p):
                        cnt = sum(1 for x in S_set if x % p == r)
                        class_counts[r] = cnt
                    min_r = min(class_counts, key=lambda r: class_counts[r])
                    violations.append((p, min_r, class_counts[min_r]))
            
            if not violations:
                break
            
            # Remove the class with smallest size
            violations.sort(key=lambda x: x[2])
            p_sel, r_sel, cnt = violations[0]
            S_set = {x for x in S_set if x % p_sel != r_sel}
        
        return list(S_set)
    
    if time.time() < deadline - 5:
        S3 = conflict_greedy()
        if S3 and len(S3) > best_k:
            best_k = len(S3)
            best_S = S3
    
    # Final verification
    if best_S is None:
        return list(range(N + 1))
    
    k = len(best_S)
    S_set = set(best_S)
    valid = True
    for p in primes:
        if p > k:
            break
        residues_seen = set(x % p for x in S_set)
        if len(residues_seen) == p:
            valid = False
            break
    
    if valid:
        return best_S
    
    # Fallback: simple greedy
    points = set(range(N + 1))
    for p in primes:
        if p > len(points):
            break
        residues = {}
        for x in points:
            r = x % p
            if r not in residues:
                residues[r] = set()
            residues[r].add(x)
        if all(r in residues and len(residues[r]) > 0 for r in range(p)):
            min_idx = min(residues, key=lambda i: len(residues[i]))
            points -= residues[min_idx]
    return list(points)

# EVOLVE-BLOCK-END
