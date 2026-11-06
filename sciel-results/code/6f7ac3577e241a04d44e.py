import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def solve(moduli):
        n = len(moduli)
        if n == 0:
            return []
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
        
        # Order variables by increasing modulus (fewer choices first)
        indices = sorted(range(n), key=lambda i: moduli[i])
        
        assigned = [None] * n
        # Store the result in original order
        result = [None] * n
        
        def backtrack(k):
            if time.time() > deadline:
                return False
            if k == n:
                return True
            
            idx = indices[k]
            m = moduli[idx]
            
            # Try all residues
            for r in range(m):
                valid = True
                # Check against all previously assigned variables
                for j in range(k):
                    j_idx = indices[j]
                    g = gcds[idx][j_idx]
                    if (r - assigned[j_idx]) % g == 0:
                        valid = False
                        break
                if valid:
                    assigned[idx] = r
                    if backtrack(k + 1):
                        return True
                    assigned[idx] = None
                    
            return False

        if backtrack(0):
            for i in range(n):
                result[i] = [assigned[i], moduli[i]]
            return result
        return None

    # Candidate sets of moduli
    # 1. Even numbers from 4 to 128 (avoids 2 which causes parity conflicts with 4,6)
    moduli_1 = list(range(4, 129, 2))
    sol = solve(moduli_1)
    if sol:
        return sol

    # 2. Even numbers from 6 to 128
    moduli_2 = list(range(6, 129, 2))
    sol = solve(moduli_2)
    if sol:
        return sol

    # 3. Multiples of 6
    moduli_3 = list(range(6, 129, 6))
    sol = solve(moduli_3)
    if sol:
        return sol

    # 4. Multiples of 4
    moduli_4 = list(range(4, 129, 4))
    sol = solve(moduli_4)
    if sol:
        return sol

    # Fallback
    return [[1, 2]]
