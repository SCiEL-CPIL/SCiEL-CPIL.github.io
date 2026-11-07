import time
import random

# Deadline guard
_START_TIME = time.time()
DEADLINE = 85.0  # Safety margin for 90s limit

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _orthogonal(a, b):
    n = len(a)
    seen = set()
    for i in range(n):
        ra = a[i]
        rb = b[i]
        for j in range(n):
            p = (ra[j], rb[j])
            if p in seen:
                return False
            seen.add(p)
    return True

def construct():
    start = time.time()
    n = 10
    
    # Base square: Group table of Z_10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second square orthogonal to L1
    # Use random isotopes of L1
    # Isotope: L[i][j] = sym[L1[row_perm[i]][col_perm[j]]]
    
    rng = random.Random(42)
    
    # We will try a few thousand random isotopes
    # This is fast enough
    found = None
    attempts = 0
    
    while time.time() - start < DEADLINE and attempts < 5000:
        attempts += 1
        
        # Random permutations
        row_perm = list(range(n))
        col_perm = list(range(n))
        sym_perm = list(range(n))
        
        rng.shuffle(row_perm)
        rng.shuffle(col_perm)
        rng.shuffle(sym_perm)
        
        # Construct isotope
        L2 = [[sym_perm[L1[row_perm[i]][col_perm[j]]] for j in range(n)] for i in range(n)]
        
        # Check orthogonality with L1
        if _orthogonal(L1, L2):
            found = L2
            break
            
    # If found, return both
    if found:
        # Ensure pure Python ints
        res1 = [[int(x) for x in row] for row in L1]
        res2 = [[int(x) for x in row] for row in found]
        return [res1, res2]
    else:
        # Fallback: return just L1
        res1 = [[int(x) for x in row] for row in L1]
        return [res1]
