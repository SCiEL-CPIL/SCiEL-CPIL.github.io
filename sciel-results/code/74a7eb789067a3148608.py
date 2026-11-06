import math
import itertools

def construct():
    def solve_for_prime(p):
        max_m = 128
        moduli = [p * k for k in range(1, max_m // p + 1)]
        n = len(moduli)
        if n == 0:
            return []
        
        # We want to assign residues r_i for each modulus m_i
        # such that r_i != r_j mod gcd(m_i, m_j)
        
        # Order moduli by value (smallest first) to constrain early
        # Actually, moduli are already sorted if we iterate k=1..N
        
        # Backtracking search
        # To keep it fast, we use a greedy approach with limited backtracking
        # or just greedy if it works
        
        residues = [None] * n
        
        def is_compatible(i, r_i):
            for j in range(i):
                if residues[j] is None:
                    continue
                m_i = moduli[i]
                m_j = moduli[j]
                g = math.gcd(m_i, m_j)
                if r_i % g == residues[j] % g:
                    return False
            return True

        def backtrack(i):
            if i == n:
                return True
            m_i = moduli[i]
            # Try residues in a smart order? 
            # Smallest residues are often fine.
            for r in range(m_i):
                if is_compatible(i, r):
                    residues[i] = r
                    if backtrack(i + 1):
                        return True
                    residues[i] = None
            return False

        # Optimization: If n is large, full backtracking might be slow.
        # But n <= 64 for p=2.
        # Let's add a timeout or limit backtracking depth?
        # For p=2, n=64.
        # For p=5, n=25.
        # For p=11, n=11.
        
        # Let's try a greedy pass first.
        def greedy():
            for i in range(n):
                m_i = moduli[i]
                found = False
                for r in range(m_i):
                    if is_compatible(i, r):
                        residues[i] = r
                        found = True
                        break
                if not found:
                    return False
            return True

        if greedy():
            return [[residues[i], moduli[i]] for i in range(n)]
        
        # If greedy fails, try backtracking with a node limit
        nodes_visited = 0
        limit = 100000
        
        def backtrack_limited(i):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > limit:
                return False
            if i == n:
                return True
            m_i = moduli[i]
            for r in range(m_i):
                if is_compatible(i, r):
                    residues[i] = r
                    if backtrack_limited(i + 1):
                        return True
                    residues[i] = None
            return False

        if backtrack_limited(0):
            return [[residues[i], moduli[i]] for i in range(n)]
            
        return []

    best_solution = []
    for p in [2, 3, 5, 7, 11, 13]:
        sol = solve_for_prime(p)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    return best_solution
