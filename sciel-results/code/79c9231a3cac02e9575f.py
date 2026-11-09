import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, aim to finish by 50s
    deadline = start_time + 50.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds for O(1) lookup
    gcd_matrix = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def is_compatible(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m', r') in selected.
        Disjoint condition: (r - r') % gcd(m, m') != 0
        """
        for m_p, r_p in selected:
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue(selected, m):
        """
        Find any r in [0, m-1] compatible with selected.
        """
        if not selected:
            return 0
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    best_solution = []
    
    # Pre-compute properties for heuristics
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    div_count = {m: count_divisors(m) for m in moduli}
    
    # Generate orderings
    orderings = []
    
    # 1. Deterministic Heuristics
    # Primes first (sorted by value), then composites sorted by value
    orderings.append(primes + composites)
    # Composites first, then primes
    orderings.append(composites + primes)
    # Sorted by divisor count (fewer divisors first - primes are 2)
    orderings.append(sorted(moduli, key=lambda x: div_count[x]))
    # Sorted by divisor count (more divisors first)
    orderings.append(sorted(moduli, key=lambda x: -div_count[x]))
    # Sorted by value
    orderings.append(sorted(moduli))
    # Sorted by value descending
    orderings.append(sorted(moduli, reverse=True))
    
    # Run deterministic ones first
    for order in orderings:
        if time.time() > deadline:
            break
        selected = []
        for m in order:
            if (len(selected) % 10 == 0) and time.time() > deadline:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
        if len(selected) > len(best_solution):
            best_solution = selected
            
    # New: CSP Solver for specific high-potential sets
    def solve_csp(moduli_set):
        """
        Backtracking search with MRV and Forward Checking.
        Returns list of (m, r) or None.
        """
        n = len(moduli_set)
        if n == 0:
            return []
        
        # Precompute pairs and gcds for this set
        # indices 0..n-1
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = gcd_matrix.get((moduli_set[i], moduli_set[j]), math.gcd(moduli_set[i], moduli_set[j]))
                gcds[i][j] = g
                gcds[j][i] = g
        
        # Check if any pair has gcd 1. If so, CSP is impossible.
        for i in range(n):
            for j in range(i+1, n):
                if gcds[i][j] == 1:
                    return None

        domains = [set(range(moduli_set[i])) for i in range(n)]
        assignment = [None] * n
        unassigned = list(range(n))
        
        # MRV: pick variable with smallest domain
        # FC: update domains of neighbors
        
        def backtrack(unassigned, domains, assignment):
            if not unassigned:
                return True
            
            # MRV
            min_idx = -1
            min_size = float('inf')
            for idx in unassigned:
                if len(domains[idx]) < min_size:
                    min_size = len(domains[idx])
                    min_idx = idx
            
            if min_size == 0:
                return False
            
            # Try values
            # Heuristic: try values that conflict least? Or just random?
            # Let's try random to avoid bias, or linear.
            # Linear is fine for small domains.
            vals = list(domains[min_idx])
            random.shuffle(vals)
            
            for r in vals:
                # Check consistency with already assigned
                consistent = True
                for j in range(len(assignment)):
                    if assignment[j] is not None:
                        g = gcds[min_idx][j]
                        if (r - assignment[j]) % g == 0:
                            consistent = False
                            break
                if not consistent:
                    continue
                
                # Assign
                assignment[min_idx] = r
                new_unassigned = [u for u in unassigned if u != min_idx]
                
                # Forward Checking: filter domains of neighbors
                new_domains = [set(d) for d in domains] # Copy
                # Optimization: only copy if necessary? n is small (~63). Copy is fast.
                fc_failed = False
                for u in new_unassigned:
                    g = gcds[min_idx][u]
                    # Filter domains[u]
                    # Remove r' such that (r' - r) % g == 0
                    # i.e., r' % g == r % g
                    target = r % g
                    # We can filter by checking (x - r) % g != 0
                    new_domain_u = set()
                    for x in domains[u]:
                        if (x - r) % g != 0:
                            new_domain_u.add(x)
                    if not new_domain_u:
                        fc_failed = True
                        break
                    new_domains[u] = new_domain_u
                
                if not fc_failed:
                    if backtrack(new_unassigned, new_domains, assignment):
                        return True
                
                # Undo
                assignment[min_idx] = None
            
            return False

        # Start
        if backtrack(unassigned, domains, assignment):
            return [(moduli_set[i], assignment[i]) for i in range(n)]
        return None

    # Try CSP on specific sets
    candidate_sets = []
    evens = [m for m in moduli if m % 2 == 0]
    candidate_sets.append(evens)
    
    threes = [m for m in moduli if m % 3 == 0]
    candidate_sets.append(threes)
    
    # Multiples of 5
    fives = [m for m in moduli if m % 5 == 0]
    candidate_sets.append(fives)
    
    # Mixed: Evens + Multiples of 3 that are also even? (Just evens)
    # What about numbers divisible by 2 or 3?
    # 2 and 3 are incompatible. So we can't mix them freely.
    # But 6 is divisible by both.
    # If we pick 6, we can pick 2? gcd(6,2)=2. Yes.
    # If we pick 6, we can pick 3? gcd(6,3)=3. Yes.
    # If we pick 2 and 3? gcd(2,3)=1. No.
    # So we can pick a set of evens, and a set of multiples of 3, as long as
    # every even in E is compatible with every multiple of 3 in T.
    # gcd(e, t) must be > 1.
    # If e is even and t is multiple of 3.
    # If e is not multiple of 3, and t is not multiple of 2, gcd(e,t) could be 1.
    # Example: e=2, t=3. gcd=1.
    # So we can only mix if they share factors.
    # This is complex. Stick to single factor sets for CSP.
    
    for c_set in candidate_sets:
        if time.time() > deadline - 5.0:
            break
        # Check if size is promising
        if len(c_set) <= len(best_solution):
            continue
            
        # Run CSP
        sol = solve_csp(c_set)
        if sol and len(sol) > len(best_solution):
            best_solution = sol

    # Now run randomized restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        
        r_choice = random.random()
        if r_choice < 0.6:
            comp = composites.copy()
            random.shuffle(comp)
            order = primes + comp
        elif r_choice < 0.8:
            comp = composites.copy()
            random.shuffle(comp)
            pri = primes.copy()
            random.shuffle(pri)
            order = comp + pri
        else:
            order = moduli.copy()
            random.shuffle(order)
            
        selected = []
        for m in order:
            if (len(selected) % 10 == 0) and time.time() > deadline - 5.0:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
                
        if len(selected) > len(best_solution):
            best_solution = selected
            
    result = [[r, m] for m, r in best_solution]
    if not result:
        return [[0, 2]]
        
    return result
