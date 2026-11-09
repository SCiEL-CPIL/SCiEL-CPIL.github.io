import time
import sys

def construct():
    n = 10
    
    # MOLS of order 5
    # L_k[i][j] = (i + (k+1)*j) % 5
    # k=0: i+j
    # k=1: i+2j
    # k=2: i+3j
    # k=3: i+4j
    m = 5
    mols_m = []
    for k in range(m - 1):
        sq = [[(i + (k + 1) * j) % m for j in range(m)] for i in range(m)]
        mols_m.append(sq)
        
    # Lift to order 10
    # L'_k[i][j] = 2 * L_k[i % m][j % m] + ((i // m) + (j // m)) % 2
    mols_n = []
    for sq in mols_m:
        new_sq = []
        for i in range(n):
            row = []
            for j in range(n):
                i1 = i // m
                i2 = i % m
                j1 = j // m
                j2 = j % m
                val = 2 * sq[i2][j2] + ((i1 + j1) % 2)
                row.append(val)
            new_sq.append(row)
        mols_n.append(new_sq)
        
    # Validation
    def is_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            col = [sq[k][i] for k in range(n)]
            if set(col) != set(range(n)):
                return False
        return True

    def is_orthogonal(sq_a, sq_b):
        seen_pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq_a[i][j], sq_b[i][j])
                if pair in seen_pairs:
                    return False
                seen_pairs.add(pair)
        return True
        
    # Check all
    valid_sqs = []
    for sq in mols_n:
        if not is_latin(sq):
            continue
        # Check orthogonality with existing valid squares
        orthogonal = True
        for v_sq in valid_sqs:
            if not is_orthogonal(sq, v_sq):
                orthogonal = False
                break
        if orthogonal:
            valid_sqs.append(sq)
            
    if not valid_sqs:
        # Fallback to single square
        sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [sq1]
        
    # Ensure types are pure Python lists and ints
    result = [
        [[int(val) for val in row] for row in sq]
        for sq in valid_sqs
    ]
    
    return result
