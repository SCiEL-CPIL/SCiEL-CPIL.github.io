"""
Rank-minimal (or near-minimal) exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy:
- Use tensor product and sum decompositions recursively.
- Base cases:
    * If any dimension is 1: trivial decomposition (rank = product of the other two).
    * (2,2,2): Strassen's algorithm (rank 7).
- Recursive cases:
    * Split one dimension into sum of two parts, use sum of decompositions.
    * Tensor product: M_{n1*n2, m1*m2, p1*p2} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2}.

For 4x5x6:
  Use tensor product: 4 = 2*2, 5 = 5*1, 6 = 3*2
  So M_{4,5,6} = M_{2,5,3} ⊗ M_{2,1,2}
  Rank(M_{2,5,3}) * Rank(M_{2,1,2})
  
  M_{2,1,2}: trivial, rank = 2*2 = 4
  M_{2,5,3}: split m=5 into 2+3
    M_{2,2,3} + M_{2,3,3}
    M_{2,2,3}: split p=3 into 2+1 -> M_{2,2,2} + M_{2,2,1}, rank = 7 + 4 = 11
    M_{2,3,3}: split p=3 into 2+1 -> M_{2,3,2} + M_{2,3,1}
      M_{2,3,2}: symmetric to M_{2,2,3} in sense that we can split m=3 into 2+1
        M_{2,2,2} + M_{2,1,2}, rank = 7 + 4 = 11
      M_{2,3,1}: trivial, rank = 2*3 = 6
    So M_{2,3,3} rank = 11 + 6 = 17
  So M_{2,5,3} rank = 11 + 17 = 28
  Total rank = 28 * 4 = 112
  
  Hmm, that's not great. Let me reconsider.
  
  Alternative: 4 = 4*1, 5 = 5*1, 6 = 6*1 doesn't help.
  
  Let me try: 4 = 2+2, so M_{4,5,6} = M_{2,5,6} + M_{2,5,6}
  That's just doubling, not helpful.
  
  Better: Use tensor product differently.
  4 = 2*2, 5 = 5*1, 6 = 3*2
  M_{4,5,6} = M_{2,5,3} ⊗ M_{2,1,2}
  
  Or: 4 = 4*1, 5 = 2*? No.
  
  Let me try: 6 = 3*2, so M_{4,5,6} = M_{4,5,3} ⊗ M_{1,1,2}
  M_{1,1,2} is trivial, rank = 1*2 = 2
  M_{4,5,3}: split n=4 into 2+2
    M_{2,5,3} + M_{2,5,3} = 2 * Rank(M_{2,5,3}) = 2*28 = 56
  Total rank = 56 * 2 = 112. Same.
  
  Let me try: M_{4,5,6} = M_{2,5,3} ⊗ M_{2,1,2}
  Rank = 28 * 4 = 112.
  
  What if I use a different decomposition for M_{2,5,3}?
  
  M_{2,5,3}: split m=5 into 3+2
    M_{2,3,3} + M_{2,2,3}
    Same as before: 17 + 11 = 28.
  
  What about using tensor product for M_{2,5,3}?
  5 = 5*1, so M_{2,5,3} = M_{2,5,3} ⊗ M_{1,1,1}, trivial.
  
  5 = ? Can't factor nicely.
  
  Let me try: M_{2,5,3} = M_{2,1,3} + M_{2,4,3}
    M_{2,1,3}: trivial, rank = 2*3 = 6
    M_{2,4,3}: split m=4 into 2+2
      M_{2,2,3} + M_{2,2,3} = 2*11 = 22
    Total: 6 + 22 = 28. Same.
  
  It seems 28 is hard to beat for M_{2,5,3}.
  
  What if I use a completely different top-level decomposition?
  
  M_{4,5,6}: 
  Option 1: 4=2*2, 5=5*1, 6=3*2 -> rank 28*4 = 112
  Option 2: 4=4*1, 5=5*1, 6=6*1 -> trivial, rank 4*5*6 = 120
  Option 3: 4=2+2, 5=5, 6=6 -> 2 * Rank(M_{2,5,6})
    M_{2,5,6}: split p=6 into 3+3
      M_{2,5,3} + M_{2,5,3} = 2*28 = 56
    Total: 2*56 = 112.
  
  Option 4: 4=2*2, 5=2*? No.
  
  Let me try: M_{4,5,6} = M_{2,2,3} ⊗ M_{2,? , ?}
  4 = 2*2, 5 = 2*2.5 no.
  
  I think 112 is the best I can do with this approach. But let me see if there's a better way.
  
  Actually, let me reconsider the tensor product structure.
  
  M_{n,m,p} ⊗ M_{n',m',p'} has rank R(n,m,p) * R(n',m',p')
  
  For M_{4,5,6}:
  - 4 = 2*2, 5 = 5*1, 6 = 3*2: M_{2,5,3} ⊗ M_{2,1,2}, rank 28*4 = 112
  - 4 = 2*2, 5 = 1*5, 6 = 2*3: M_{2,1,2} ⊗ M_{2,5,3}, same
  - 4 = 1*4, 5 = 1*5, 6 = 1*6: trivial
  
  What if I use: 4 = 2*2, 5 = 5*1, 6 = 6*1
  M_{2,5,6} ⊗ M_{2,1,1}
  M_{2,1,1}: trivial, rank = 2*1 = 2
  M_{2,5,6}: split p=6 into 3+3
    M_{2,5,3} + M_{2,5,3} = 56
  Total: 56*2 = 112.
  
  I'm consistently getting 112. Let me just implement this.
  
  Actually, wait. Let me try a different approach for M_{2,5,3}.
  
  M_{2,5,3}: Can I use tensor product?
  5 = 5*1, so no benefit.
  3 = 3*1, so no.
  2 = 2*1, so no.
  
  What if I split differently?
  M_{2,5,3} = M_{2,5,1} + M_{2,5,2}
    M_{2,5,1}: trivial, rank = 2*5 = 10
    M_{2,5,2}: split m=5 into 3+2
      M_{2,3,2} + M_{2,2,2}
      M_{2,3,2}: split m=3 into 2+1
        M_{2,2,2} + M_{2,1,2} = 7 + 4 = 11
      So M_{2,5,2} = 11 + 7 = 18
    Total: 10 + 18 = 28.
  
  Still 28.
  
  Let me just implement the 112-rank decomposition.
"""

import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _trivial(n, m, p):
    """Trivial decomposition for cases where one dimension is 1."""
    if n == 1:
        # M_{1,m,p}: rank = m*p
        rank = m * p
        U = [[0] * rank for _ in range(1 * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * 1)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W
    elif m == 1:
        # M_{n,1,p}: rank = n*p
        rank = n * p
        U = [[0] * rank for _ in range(n * 1)]
        V = [[0] * rank for _ in range(1 * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W
    elif p == 1:
        # M_{n,m,1}: rank = n*m
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * 1)]
        W = [[0] * rank for _ in range(1 * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W
    else:
        # Should not happen if called correctly
        raise ValueError("Trivial decomposition requires one dimension to be 1")


def _strassen():
    """Strassen's algorithm for M_{2,2,2}, rank 7."""
    # The 7 terms of Strassen's algorithm
    # M_{2,2,2}: C = A * B where A is 2x2, B is 2x2, C is 2x2
    # U: 2*2 = 4 rows, V: 2*2 = 4 rows, W: 2*2 = 4 rows
    # Each column corresponds to one of the 7 products
    
    # Strassen's 7 products:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # U[i*m+j, c] where i,j index A, c is the column (product index)
    # V[j*p+k, c] where j,k index B, c is the column
    # W[k*n+i, c] where k,i index C, c is the column
    
    # A indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # P1 = (A11+A22)*(B11+B22)
    # U: A11 (0,0) and A22 (1,1) get +1
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    # V: B11 (0,0) and B22 (1,1) get +1
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # W: C11 (0,0) gets +1, C22 (1,1) gets +1
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # P2 = (A21+A22)*B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21
    W[3][1] = -1  # C22
    
    # P3 = A11*(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12
    V[3][2] = -1  # B22
    W[1][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # P4 = A22*(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21
    V[0][3] = -1  # B11
    W[0][3] = 1  # C11
    W[2][3] = 1  # C21
    
    # P5 = (A11+A12)*B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    W[0][4] = -1  # C11
    W[1][4] = 1  # C12
    
    # P6 = (A21-A11)*(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1  # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22
    
    # P7 = (A12-A22)*(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1  # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11
    W[3][6] = 1  # C22
    
    return U, V, W


def _sum_decomp(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (sum)."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 + r2
    
    U = []
    for i in range(len(U1)):
        row = U1[i] + U2[i]
        U.append(row)
    
    V = []
    for i in range(len(V1)):
        row = V1[i] + V2[i]
        V.append(row)
    
    W = []
    for i in range(len(W1)):
        row = W1[i] + W2[i]
        W.append(row)
    
    return U, V, W


def _tensor_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Tensor product of two decompositions.
    M_{n1*n2, m1*m2, p1*p2} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2}
    """
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each pair of columns (c1, c2), create a new column c = c1 * r2 + c2
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            # A index: (i1*n2 + i2, j1*m2 + j2)
                            a_idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            val = U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                            if val != 0:
                                U[a_idx][c] += val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # B index: (j1*m2 + j2, k1*p2 + k2)
                            b_idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            val = V1[j1 * p1 + k1][c1] * V2[j2 * p2 + k2][c2]
                            if val != 0:
                                V[b_idx][c] += val
            
            for i1 in range(n1):
                for k1 in range(p1):
                    for i2 in range(n2):
                        for k2 in range(p2):
                            # C index: (k1*p2 + k2, i1*n2 + i2)
                            c_idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            val = W1[k1 * n1 + i1][c1] * W2[k2 * n2 + i2][c2]
                            if val != 0:
                                W[c_idx][c] += val
    
    return U, V, W


def _decompose(n, m, p):
    """Recursively decompose M_{n,m,p}."""
    if n == 1 or m == 1 or p == 1:
        return _trivial(n, m, p)
    
    if n == 2 and m == 2 and p == 2:
        return _strassen()
    
    # Try tensor product decompositions
    # Try to factor n, m, p into products
    
    best = None
    best_rank = float('inf')
    
    # Try all possible tensor product splits
    for n1 in range(1, n + 1):
        if n % n1 != 0:
            continue
        n2 = n // n1
        for m1 in range(1, m + 1):
            if m % m1 != 0:
                continue
            m2 = m // m1
            for p1 in range(1, p + 1):
                if p % p1 != 0:
                    continue
                p2 = p // p1
                
                U1, V1, W1 = _decompose(n1, m1, p1)
                U2, V2, W2 = _decompose(n2, m2, p2)
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                rank = r1 * r2
                
                if rank < best_rank:
                    best_rank = rank
                    best = (n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2)
    
    if best is not None:
        n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2 = best
        return _tensor_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2)
    
    # Try sum decompositions
    # Split one dimension
    best = None
    best_rank = float('inf')
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        rank = r1 + r2
        if rank < best_rank:
            best_rank = rank
            best = ('n', n1, n2, U1, V1, W1, U2, V2, W2)
    
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        rank = r1 + r2
        if rank < best_rank:
            best_rank = rank
            best = ('m', m1, m2, U1, V1, W1, U2, V2, W2)
    
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        rank = r1 + r2
        if rank < best_rank:
            best_rank = rank
            best = ('p', p1, p2, U1, V1, W1, U2, V2, W2)
    
    if best is not None:
        kind, a, b, U1, V1, W1, U2, V2, W2 = best
        return _sum_decomp(U1, V1, W1, U2, V2, W2)
    
    # Fallback: schoolbook
    return _trivial(n, m, p) if (n == 1 or m == 1 or p == 1) else None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    U, V, W = _decompose(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
