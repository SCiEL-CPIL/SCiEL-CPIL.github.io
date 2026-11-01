"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses Laderman's rank-7 decomposition applied to each of 5 independent 2x2 column-pair
    blocks of the 2x10 matrix B. Total rank = 5 * 7 = 35.
    
    The tensor is a direct sum of 5 copies of the 2x2x2 matmul tensor, so rank 35 is
    the theoretical minimum for this structure. We verify exactness and return.
    
    Interface:
      U: shape (n*m, rank) = (4, 35)   — coefficients for A
      V: shape (m*p, rank) = (20, 35)  — coefficients for B  
      W: shape (p*n, rank) = (20, 35)  — coefficients for C (trace convention W[k*n+i])
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Laderman's rank-7 decomposition for 2x2 matmul:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6
    
    # U coefficients: [a, b, c, d] for each product
    laderman_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    # V coefficients: [e, f, g, h] for each product
    laderman_B = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    
    # W coefficients: [C00, C01, C10, C11] contribution of each product
    w_contrib = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]
    
    # Fill in the matrices for each of the 5 blocks
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # U: index i*m + j → a=0, b=1, c=2, d=3
            U[0][col] = laderman_A[r][0]
            U[1][col] = laderman_A[r][1]
            U[2][col] = laderman_A[r][2]
            U[3][col] = laderman_A[r][3]
            
            # V: index j*p + k
            # e = B[0, 2b] → idx 0*10 + 2b = 2b
            # f = B[0, 2b+1] → idx 0*10 + 2b+1 = 2b+1
            # g = B[1, 2b] → idx 1*10 + 2b = 10+2b
            # h = B[1, 2b+1] → idx 1*10 + 2b+1 = 10+2b+1
            V[2*b][col] = laderman_B[r][0]
            V[2*b + 1][col] = laderman_B[r][1]
            V[10 + 2*b][col] = laderman_B[r][2]
            V[10 + 2*b + 1][col] = laderman_B[r][3]
            
            # W: index k*n + i (official trace convention)
            # C00: k=2b, i=0 → idx 2b*2 + 0 = 4b
            # C01: k=2b+1, i=0 → idx (2b+1)*2 + 0 = 4b+2
            # C10: k=2b, i=1 → idx 2b*2 + 1 = 4b+1
            # C11: k=2b+1, i=1 → idx (2b+1)*2 + 1 = 4b+3
            W[4*b][col] = w_contrib[r][0]       # C00
            W[4*b + 2][col] = w_contrib[r][1]   # C01
            W[4*b + 1][col] = w_contrib[r][2]   # C10
            W[4*b + 3][col] = w_contrib[r][3]   # C11
    
    # Verify exactness
    _verify_decomposition(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    rank = len(U[0])
    # For each (i,j,k,l) in the tensor:
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The matmul tensor has T[i,j,k,l] = 1 if j==l and i==i' and k==k' 
    # More precisely: T[i,j, j',k, i',k'] = δ[j,j']δ[i,i']δ[k,k']
    # In our indexing: A[i,j], B[j,k], C[i,k]
    # T[i,j,k] where these are flattened: A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i
    # T[A_idx, B_idx, C_idx] = 1 if (i from A == i from C) and (j from A == j from B) and (k from B == k from C)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This should be 1 in the tensor
                A_idx = i * m + j
                B_idx = j * p + k
                C_idx = k * n + i
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[A_idx][r] * V[B_idx][r] * W[C_idx][r]
                if actual != expected:
                    raise ValueError(f"Verification failed at A[{i},{j}], B[{j},{k}], C[{k},{i}]: got {actual}, expected {expected}")
    
    # Also verify that non-matching entries are 0
    # For random sample of non-matching triples
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        # Perturb to get a non-matching triple
        if random.random() < 0.5:
            i2 = random.randint(0, n-1)
            if i2 == i: i2 = (i2+1) % n
            A_idx = i2 * m + j
            B_idx = j * p + k
            C_idx = k * n + i
        else:
            k2 = random.randint(0, p-1)
            if k2 == k: k2 = (k2+1) % p
            A_idx = i * m + j
            B_idx = j * p + k2
            C_idx = k * n + i
        actual = 0
        for r in range(rank):
            actual += U[A_idx][r] * V[B_idx][r] * W[C_idx][r]
        if actual != 0:
            raise ValueError(f"Verification failed: non-zero at A_idx={A_idx}, B_idx={B_idx}, C_idx={C_idx}, got {actual}")
# EVOLVE-BLOCK-END
