import time
import math

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # All even moduli 2..128
    all_moduli = [m for m in range(2, 129) if m % 2 == 0]
    all_moduli.sort()
    
    n_full = len(all_moduli)
    
    # Precompute gcds for the full set
    gcds_full = [[0]*n_full for _ in range(n_full)]
    for i in range(n_full):
        for j in range(i+1, n_full):
            gcds_full[i][j] = math.gcd(all_moduli[i], all_moduli[j])
            gcds_full[j][i] = gcds_full[i][j]
            
    initial_domains_full = [(1 << m) - 1 for m in all_moduli]

    def solve_subset(indices):
        """
        Solves the CSP for the given list of indices into all_moduli.
        Returns a dict {index: residue} or None.
        """
        n = len(indices)
        if n == 0:
            return {}
        
        # Map original index to local index
        # We need to rebuild gcds and domains for the subset?
        # Or just use the full ones and skip?
        # Rebuilding is cleaner.
        
        moduli = [all_moduli[i] for i in indices]
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
                gcds[j][i] = gcds[i][j]
                
        domains = [(1 << m) - 1 for m in moduli]
        assignment = [-1] * n
        
        # MRV Heuristic: track domain sizes
        # We will select the variable with the smallest non-zero domain size
        
        def backtrack():
            if time.time() - start_time > time_limit:
                return None
                
            # Find unassigned variable with smallest domain
            min_size = float('inf')
            min_idx = -1
            for i in range(n):
                if assignment[i] == -1:
                    size = bin(domains[i]).count('1')
                    if size == 0:
                        return None # Conflict
                    if size < min_size:
                        min_size = size
                        min_idx = i
                        
            if min_idx == -1:
                return assignment.copy() # All assigned
                
            m = moduli[min_idx]
            mask = domains[min_idx]
            
            # Value ordering: Try values that reduce neighbor domains most?
            # For simplicity, just iterate.
            while mask:
                lsb = mask & -mask
                r = lsb.bit_length() - 1
                
                assignment[min_idx] = r
                
                # Propagate
                new_domains = domains[:]
                valid = True
                for j in range(n):
                    if assignment[j] == -1 and j != min_idx:
                        g = gcds[min_idx][j]
                        if g > 1:
                            m_j = moduli[j]
                            # Forbidden residues in m_j
                            start = r % g
                            forbidden = 0
                            for k in range(start, m_j, g):
                                forbidden |= (1 << k)
                            new_domains[j] &= ~forbidden
                            if new_domains[j] == 0:
                                valid = False
                                break
                                
                if valid:
                    # Recurse
                    domains_backup = domains
                    domains = new_domains
                    res = backtrack()
                    domains = domains_backup
                    if res is not None:
                        return res
                else:
                    # Undo propagation? We already made a copy, so just discard new_domains
                    pass
                    
                assignment[min_idx] = -1
                mask ^= lsb
                
            return None
            
        return backtrack()

    # Try full set first
    indices = list(range(n_full))
    sol = solve_subset(indices)
    if sol is not None:
        # Map back to residues and moduli
        result = []
        for i in range(len(indices)):
            result.append([sol[i], all_moduli[indices[i]]])
        return result
        
    # If full set fails, try removing one modulus at a time
    # We try removing the last ones first (largest moduli) as they might be more flexible?
    # Or first ones?
    # Let's try removing one and retrying.
    # To save time, we limit the number of retries.
    for k in range(n_full - 1, 0, -1):
        if time.time() - start_time > time_limit - 5: # Keep 5s buffer
            break
        # Try subset of size k
        # Which k? Let's try the first k (smallest moduli)
        # Or maybe the ones with highest 2-adic valuation?
        # Let's just try the first k (smallest values)
        indices = list(range(k))
        sol = solve_subset(indices)
        if sol is not None:
            result = []
            for i in range(len(indices)):
                result.append([sol[i], all_moduli[indices[i]]])
            return result
            
    # Fallback to seed
    return [[(1<<k)-1,1<<(k+1)] for k in range(7)]
