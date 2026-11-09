"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 10
    
    # Start with a base Latin square
    def base_latin():
        return [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_latin(L):
        for i in range(n):
            if sorted(L[i]) != list(range(n)):
                return False
        for j in range(n):
            if sorted([L[i][j] for i in range(n)]) != list(range(n)):
                return False
        return True
    
    def is_orthogonal(A, B):
        seen = set()
        for i in range(n):
            for j in range(n):
                p = (A[i][j], B[i][j])
                if p in seen:
                    return False
                seen.add(p)
        return len(seen) == n * n
    
    # Generate Latin squares by permuting rows/columns/symbols of base
    def permute_latin(L, row_perm, col_perm, sym_perm):
        return [[sym_perm[L[row_perm[i]][col_perm[j]]] for j in range(n)] for i in range(n)]
    
    def random_perm():
        p = list(range(n))
        random.shuffle(p)
        return p
    
    # We'll try to build a set of mutually orthogonal Latin squares
    # Start with two known orthogonal ones, then search for more
    
    # Known orthogonal pair for order 10:
    # L1: (i+j) % 10
    # L2: (i+2j) % 10 -- but this may not be orthogonal to L1 for n=10
    
    # Let's verify: pairs (i+j, i+2j) mod 10
    # For fixed i, as j varies, we get (i+j, i+2j). 
    # Two cells (i1,j1) and (i2,j2) give same pair if:
    # i1+j1 = i2+j2 and i1+2j1 = i2+2j2
    # Subtract: j1 = j2, then i1 = i2. So they ARE orthogonal!
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2*j) % n for j in range(n)] for i in range(n)]
    
    # Verify
    assert is_latin(L1)
    assert is_latin(L2)
    assert is_orthogonal(L1, L2)
    
    result = [L1, L2]
    
    # Try to find a third square orthogonal to both
    # Use random permutations of the base square and check
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and len(result) < 3 and attempts < max_attempts:
        attempts += 1
        if time.time() > deadline:
            break
        
        rp = random_perm()
        cp = random_perm()
        sp = random_perm()
        candidate = permute_latin(L1, rp, cp, sp)
        
        # Quick check: must be orthogonal to all in result
        ok = True
        for existing in result:
            if not is_orthogonal(candidate, existing):
                ok = False
                break
        
        if ok:
            result.append(candidate)
    
    # Try a few more systematic candidates if time permits
    if time.time() < deadline and len(result) < 3:
        # Try L3 = (i + 3j) % 10
        L3 = [[(i + 3*j) % n for j in range(n)] for i in range(n)]
        if is_latin(L3) and all(is_orthogonal(L3, s) for s in result):
            result.append(L3)
    
    # Convert to proper Python ints
    final = []
    for sq in result:
        final.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return final
# EVOLVE-BLOCK-END
