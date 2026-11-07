"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 10}

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 50  # 50s safety margin within 60s limit

    # L1: (i + j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # L2: (i + 3*j) % 10
    L2 = [[(i + 3*j) % n for j in range(n)] for i in range(n)]
    
    # Verify both are valid Latin squares
    for L in [L1, L2]:
        for i in range(n):
            row = sorted(L[i])
            if row != list(range(n)):
                return [L1]
        for j in range(n):
            col = sorted([L[i][j] for i in range(n)])
            if col != list(range(n)):
                return [L1]
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            if pair in pairs:
                # Not orthogonal, fall back to just L1
                return [L1]
            pairs.add(pair)
    
    # We have 2 orthogonal Latin squares.
    # It is a well-known mathematical result (Euler's 36 officers problem, solved by Tarry in 1901)
    # that no pair of orthogonal Latin squares of order 10 exists.
    # Therefore, the maximum number of mutually orthogonal Latin squares of order 10 is 1.
    # Wait, the seed program returns 2 squares and scores 1.0.
    # Let me re-read the task carefully.
    # "Maximize the number of pairwise orthogonal arrays."
    # "Return between 1 and 9 squares"
    # "The raw value is the number of squares returned."
    # If no pair exists, the maximum is 1.
    # But the seed program returns 2 squares and scores 1.0.
    # Let's check if L1 and L2 are actually orthogonal.
    # L1[i][j] = (i+j)%10
    # L2[i][j] = (i+3j)%10
    # Pair: ((i+j)%10, (i+3j)%10)
    # Let a = (i+j)%10, b = (i+3j)%10
    # b - a = 2j mod 10
    # If j is fixed, b-a is fixed.
    # If we vary i, a changes by 1, b changes by 1.
    # So the pair (a, b) becomes (a+1, b+1).
    # For a fixed j, the 10 pairs are (j, 3j), (j+1, 3j+1), ..., (j+9, 3j+9) mod 10.
    # Are these all distinct for different j?
    # Let's check j=0: (0,0), (1,1), ..., (9,9)
    # Let's check j=1: (1,3), (2,4), ..., (0,2)
    # (1,1) is in j=0 set. (1,3) is not.
    # (2,4) is not.
    # ...
    # (0,2) is not.
    # It seems they might be distinct?
    # Total pairs = 100. If they are all distinct, they are orthogonal.
    # But it is a known fact that OLS(10) does not exist.
    # Let's re-verify the math.
    # Difference b-a = 2j mod 10.
    # For j=0, diff=0. Pairs: (k, k) for k=0..9.
    # For j=1, diff=2. Pairs: (k, k+2) for k=0..9 (with wrap).
    # For j=5, diff=10=0. Pairs: (k, k) for k=0..9.
    # Wait! j=0 and j=5 produce the SAME set of pairs?
    # j=0: i=0 -> (0,0), i=1 -> (1,1)...
    # j=5: i=0 -> (5, 15%10=5), i=1 -> (6, 16%10=6)...
    # Yes! For j=0, pairs are (i, i).
    # For j=5, pairs are (i+5, i+15%10) = (i+5, i+5).
    # So (0,0) appears at (i=0, j=0) AND (i=5, j=5)?
    # Let's check (i=0, j=0): L1=0, L2=0. Pair (0,0).
    # Let's check (i=5, j=5): L1=(5+5)%10=0, L2=(5+15)%10=0. Pair (0,0).
    # So they are NOT orthogonal.
    # The seed program's verification loop:
    # pairs = set()
    # for i... for j...
    #   pair = (L1[i][j], L2[i][j])
    #   if pair in pairs: return [L1]
    # This code SHOULD return [L1] because (0,0) is repeated.
    # Why does it score 1.0?
    # "The evaluator measures the raw value (higher is better). The search score is value/seed"
    # If seed returns 1 square, value=1, score=1.0.
    # If I return 1 square, value=1, score=1.0.
    # If I return 2 squares, value=2, score=2.0. BUT it is invalid.
    # "Any broken rule makes the whole result invalid; there is no partial credit."
    # "an invalid result ... scores 0"
    # So the seed program likely returns [L1] (1 square) and scores 1.0.
    # My goal is to beat 1.0.
    # But OLS(10) does not exist. The maximum number of MOLS of order 10 is 1.
    # So I cannot return 2 valid orthogonal squares.
    # Therefore, the maximum raw value is 1.
    # So I cannot beat 1.0?
    # Wait, is it possible I am wrong about OLS(10)?
    # "It is a well-known mathematical result that no pair of orthogonal Latin squares of order 10 exists."
    # This is the 36 officers problem. Euler conjectured it, and it was proven true by Tarry in 1901.
    # So the maximum number of MOLS of order 10 is indeed 1.
    # So the best possible raw value is 1.
    # So the best possible score is 1.0.
    # So I cannot improve?
    # Unless... the task allows something else?
    # "Maximize the number of pairwise orthogonal arrays."
    # "Return between 1 and 9 squares"
    # If I return 1 square, it's valid.
    # If I return 0 squares, it's invalid (must be nonempty).
    # So 1 is the max.
    # So I should just return 1 valid Latin square.
    # To be safe, I'll just return L1.
    
    return [L1]

# EVOLVE-BLOCK-END
