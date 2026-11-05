import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline
    
    best_poly = [1] * 12
    best_score = 23  # Seed score
    
    # Helper to count nonzero terms in square
    def count_square_terms(p):
        # p is list of coefficients, index is degree
        # We need to handle sparse representation or dense?
        # Dense is easier for small degrees.
        # Max degree 40.
        # Square max degree 80.
        # Use list of length 81.
        sq = [0] * 81
        for i in range(len(p)):
            if p[i] != 0:
                for j in range(len(p)):
                    if p[j] != 0:
                        sq[i+j] += p[i] * p[j]
        return sum(1 for x in sq if x != 0)
    
    # Try the known candidate with 23 terms (seed)
    # Try to find better
    
    # Strategy: Random search on exponents and coefficients
    # Exponents must be distinct, in [0, 40]
    # Coefficients in [-64, 64]
    
    # We can also try structured families
    # e.g. P(x) = A(x^k)
    
    # Let's try a few specific candidates first
    candidates = []
    
    # Candidate 1: Seed
    candidates.append([1]*12)
    
    # Candidate 2: 12 terms at even powers
    p = [0]*23
    for i in range(12):
        p[2*i] = 1
    candidates.append(p)
    
    # Candidate 3: Random sparse
    for _ in range(1000):
        if time.time() > deadline:
            break
        exps = random.sample(range(41), 12)
        exps.sort()
        p = [0]*41
        for e in exps:
            p[e] = random.choice([-1, 1])
        candidates.append(p)
        
    # Evaluate candidates
    for p in candidates:
        if time.time() > deadline:
            break
        # Check constraints
        if len(p) > 41: continue
        if p[-1] == 0:
            # trim trailing zeros? 
            # "nonzero final coefficient" means last element of list must be non-zero?
            # "Return coefficient list in ascending degree, length<=41, nonzero final coefficient"
            # This implies the list should not have trailing zeros.
            while p and p[-1] == 0:
                p.pop()
            if not p: continue
        # Check exactly 12 nonzero terms
        if sum(1 for x in p if x != 0) != 12:
            continue
        # Check coefficients in [-64, 64]
        if any(abs(x) > 64 for x in p):
            continue
            
        score = count_square_terms(p)
        if score < best_score:
            best_score = score
            best_poly = p[:]
            
    # If we found something better, return it
    # Ensure final format
    while best_poly and best_poly[-1] == 0:
        best_poly.pop()
        
    return best_poly
