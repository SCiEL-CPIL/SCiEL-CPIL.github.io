import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    
    MAX_M = 128
    all_moduli = list(range(2, MAX_M + 1))
    n = len(all_moduli)
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Greedy: assign residues in order of moduli.
    # For each modulus m_i, pick a residue r such that for all j < i,
    # r % gcd(m_i, m_j) != r_j % gcd(m_i, m_j).
    # If no such r exists, try a different r for some previous assignment.
    
    # Simple greedy first:
    residues = [None] * n
    
    def is_valid(i, r):
        """Check if residue r for modulus i is compatible with all assigned j < i."""
        for j in range(i):
            if residues[j] is None:
                continue
            g = gcds[i][j]
            if g > 1 and r % g == residues[j] % g:
                return False
        return True
    
    # Assign in ascending order of modulus
    for i in range(n):
        m = all_moduli[i]
        found = False
        for r in range(m):
            if is_valid(i, r):
                residues[i] = r
                found = True
                break
        if not found:
            # Try random residue
            import random
            random.seed(42)
            for _ in range(1000):
                r = random.randint(0, m - 1)
                if is_valid(i, r):
                    residues[i] = r
                    found = True
                    break
            if not found:
                # Skip this modulus
                residues[i] = None
    
    # Build result
    result = []
    for i in range(n):
        if residues[i] is not None:
            result.append([residues[i], all_moduli[i]])
    
    if not result:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result
# EVOLVE-BLOCK-END
