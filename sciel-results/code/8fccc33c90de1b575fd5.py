import time
from math import gcd
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n = len(moduli)
    
    # Precompute constraints
    constraints = [[] for _ in range(n)]
    g_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            if g > 1:
                constraints[i].append((j, g))
                constraints[j].append((i, g))
                g_matrix[i][j] = g
                g_matrix[j][i] = g

    def is_compatible(r_i, r_j, g):
        return (r_i % g) != (r_j % g)

    def ac3_restricted(domains, initial_queue):
        """
        domains: list of sets.
        initial_queue: deque of (i, j) arcs.
        Modifies domains in place.
        Returns False if domain wipeout.
        """
        queue = initial_queue
        while queue:
            i, j = queue.popleft()
            g = g_matrix[i][j]
            
            to_remove = set()
            # Optimization: If domains[j] is small, iterate over it.
            # If domains[i] is small, iterate over it.
            # Here we check if each r_i in domains[i] has support in domains[j].
            
            # If len(domains[j]) is 1, we can check faster?
            # Let's just do the double loop, domains are small.
            
            for r_i in domains[i]:
                has_support = False
                for r_j in domains[j]:
                    if is_compatible(r_i, r_j, g):
                        has_support = True
                        break
                if not has_support:
                    to_remove.add(r_i)
            
            if to_remove:
                domains[i] = domains[i] - to_remove
                if not domains[i]:
                    return False
                # Add neighbors of i (except j) to queue
                for k, _ in constraints[i]:
                    if k != j:
                        queue.append((k, i))
                        
        return True

    def solve_csp():
        domains = [set(range(moduli[i])) for i in range(n)]
        
        # Initial AC-3
        init_queue = deque()
        for i in range(n):
            for j, _ in constraints[i]:
                init_queue.append((i, j))
                
        if not ac3_restricted(domains, init_queue):
            return None
            
        def backtrack(domains):
            if time.time() > deadline:
                return None
                
            # Check if solved
            solved = True
            for d in domains:
                if len(d) > 1:
                    solved = False
                    break
            if solved:
                return [list(d)[0] for d in domains]
                
            # MRV Heuristic
            min_size = float('inf')
            var = -1
            for i in range(n):
                if 1 < len(domains[i]) < min_size:
                    min_size = len(domains[i])
                    var = i
                    
            if var == -1:
                # All domains size 1? Should have been caught above.
                # Or all size 0? Handled by AC-3.
                return [list(d)[0] for d in domains]
                
            # Try values
            # Order values by least constraining value (LCV) heuristic?
            # For now, sorted order.
            for val in sorted(domains[var]):
                new_domains = [set(d) for d in domains]
                new_domains[var] = {val}
                
                # Create queue for AC-3
                # We changed var, so neighbors k of var need to check arcs (k, var)
                q = deque([(k, var) for k, _ in constraints[var]])
                
                if not ac3_restricted(new_domains, q):
                    continue
                    
                res = backtrack(new_domains)
                if res is not None:
                    return res
                    
            return None
            
        return backtrack(domains)

    solution = solve_csp()
    
    if solution:
        # Final verification
        valid = True
        for i in range(n):
            for j, g in constraints[i]:
                if i < j:
                    if (solution[i] % g) == (solution[j] % g):
                        valid = False
                        break
            if not valid:
                break
                
        if valid:
            return [[solution[i], moduli[i]] for i in range(n)]

    # Fallback: Primes (pairwise coprime, always disjoint)
    primes = [m for m in moduli if all(m % p != 0 for p in range(2, int(m**0.5)+1))]
    return [[0, p] for p in primes]
