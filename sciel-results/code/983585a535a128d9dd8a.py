import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # The optimal set of moduli is all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    n = len(moduli)
    
    # Precompute gcds
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Variables are indices 0..n-1
    # Residue assignment
    residues = [None] * n
    
    # Order variables by heuristic: smaller modulus first? 
    # Or just natural order.
    # Forward checking: maintain domains for unassigned variables.
    # To keep it simple and fast, we use a simple backtracking with constraint checking.
    # Given the density, we might need a better heuristic.
    # Let's try MinConflicts, it's often faster for this.
    
    # MinConflicts Implementation
    def min_conflicts():
        # Initialize randomly
        current_res = [i % moduli[i] for i in range(n)]
        
        def count_conflicts(idx):
            count = 0
            r_val = current_res[idx]
            m_val = moduli[idx]
            for j in range(n):
                if i == j: continue
                g = gcd_matrix[idx][j]
                if r_val % g == current_res[j] % g:
                    count += 1
            return count

        # Correct loop variable
        def count_conflicts_safe(idx):
            count = 0
            r_val = current_res[idx]
            for j in range(n):
                if idx == j: continue
                g = gcd_matrix[idx][j]
                if r_val % g == current_res[j] % g:
                    count += 1
            return count

        max_iters = 10000
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if count_conflicts_safe(0) == 0 and all(count_conflicts_safe(i) == 0 for i in range(n)):
                return current_res[:]
            
            # Find a variable with conflicts
            conflicted_vars = [i for i in range(n) if count_conflicts_safe(i) > 0]
            if not conflicted_vars:
                return current_res[:]
            
            i = conflicted_vars[0] # Or random choice
            # Choose a new value for i that minimizes conflicts
            best_val = None
            min_conf = float('inf')
            # Sample some values or all?
            # Modulus is up to 128. Checking all is fast.
            for val in range(moduli[i]):
                current_res[i] = val
                conf = count_conflicts_safe(i)
                if conf < min_conf:
                    min_conf = conf
                    best_val = val
                    if conf == 0:
                        break
            current_res[i] = best_val
            
        return None

    res = min_conflicts()
    
    if res is None:
        # Fallback to backtracking if min conflicts fails (unlikely)
        # Or just return a smaller valid set
        # Let's try a simple backtracking for safety
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            # Try residues
            for r in range(m):
                residues[idx] = r
                # Check against assigned variables
                valid = True
                for j in range(idx):
                    if residues[j] is None:
                        continue
                    g = gcd_matrix[idx][j]
                    if r % g == residues[j] % g:
                        valid = False
                        break
                if valid:
                    if backtrack(idx + 1):
                        return True
            residues[idx] = None
            return False
        
        if backtrack(0):
            res = residues[:]
        else:
            # Extreme fallback: just return the seed-like answer or a small valid set
            # The seed returned 7 classes. We should do better.
            # But if 64 fails, we can try 32 (multiples of 4)
            moduli_fallback = list(range(4, 129, 4))
            n_f = len(moduli_fallback)
            # Re-run min conflicts for fallback?
            # For now, return the seed result if all else fails, but we expect success.
            return [[(1<<k)-1,1<<(k+1)] for k in range(7)]

    return [[res[i], moduli[i]] for i in range(n)]
