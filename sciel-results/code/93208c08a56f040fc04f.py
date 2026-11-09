N = 5000
K = 100

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(S):
    k = len(S)
    if k == 0:
        return True
    primes = _primes(k)
    for p in primes:
        residues = set()
        for v in S:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True

def construct():
    S = list(range(N + 1))
    primes_K = _primes(K)
    
    # Precompute counts for each prime and residue
    # counts[p][r] = number of elements in S with x % p == r
    counts = {}
    for p in primes_K:
        counts[p] = [0] * p
        for v in S:
            counts[p][v % p] += 1
            
    independent_set = []
    remaining = S[:]
    
    while remaining:
        # Find vertex with min degree
        # Degree approx = sum of (count[p][v%p] - 1) for p in primes_K
        # This overcounts neighbors shared by multiple primes, but is a good heuristic
        min_deg = float('inf')
        best_v = None
        
        for v in remaining:
            deg = 0
            for p in primes_K:
                deg += counts[p][v % p] - 1
            if deg < min_deg:
                min_deg = deg
                best_v = v
                
        if best_v is None:
            break
            
        independent_set.append(best_v)
        
        # Remove best_v and its neighbors from remaining
        # Neighbors are u such that exists p, u % p == best_v % p
        # We update counts and remaining
        
        # To remove neighbors efficiently:
        # A neighbor u is in remaining and shares a residue with best_v for some p in primes_K
        # We can iterate through remaining and check? O(N^2) total?
        # N=5000. 5000*5000 = 25M.
        # Inside check: loop over primes (25). 625M ops. Too slow.
        
        # Better:
        # Construct a set of "forbidden" residues for each prime?
        # No, we remove specific nodes.
        
        # Let's just filter remaining
        # new_remaining = []
        # for u in remaining:
        #    if u == best_v: continue
        #    is_neighbor = False
        #    for p in primes_K:
        #       if u % p == best_v % p:
        #          is_neighbor = True
        #          break
        #    if not is_neighbor:
        #       new_remaining.append(u)
        
        # This is O(N * |P|) per step.
        # Total O(N^2 * |P|) = 25M * 25 = 625M.
        # In Python, this might take 10-20 seconds. Acceptable.
        
        target_residues = {p: best_v % p for p in primes_K}
        
        new_remaining = []
        for u in remaining:
            if u == best_v:
                continue
            is_neighbor = False
            for p in primes_K:
                if u % p == target_residues[p]:
                    is_neighbor = True
                    break
            if not is_neighbor:
                new_remaining.append(u)
        
        # Update counts
        # We removed best_v and its neighbors.
        # It's easier to recompute counts from new_remaining if N is small?
        # Or update counts for removed nodes.
        
        # Nodes removed: best_v and all neighbors.
        # We can just recompute counts from new_remaining for simplicity?
        # O(N * |P|) per step. Same cost.
        
        for p in primes_K:
            counts[p] = [0] * p
            for v in new_remaining:
                counts[p][v % p] += 1
                
        remaining = new_remaining
        
    # Check admissibility
    if not _is_admissible(independent_set):
        # Fallback to a known admissible set
        # All even numbers? 
        # S = [i for i in range(0, N+1, 2)]
        # Check if admissible.
        # If not, return [0]
        return [0]
        
    return independent_set
