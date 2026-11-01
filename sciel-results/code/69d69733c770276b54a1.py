import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x8 matrix.
    
    The rank is 28, derived from applying Strassen's algorithm (rank 7) 
    to each of the four 2x2 blocks of the 2x8 matrix B.
    """
    n, m, p = 2, 2, 8
    num_blocks = p // m  # 4 blocks
    rank = num_blocks * 7  # 28
    
    # Dimensions of factor matrices
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 x 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # 
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    #
    # C = [[M1 + M4 - M5 + M7, M3 + M5],
    #      [M2 + M4,           M1 - M2 + M3 + M6]]
    
    # Indices for A (U)
    # A[0] = a, A[1] = b, A[2] = c, A[3] = d
    # 
    # Indices for B block j (V)
    # Block j corresponds to columns 2j and 2j+1 of B.
    # B entries in V are at indices:
    # e: row 0, col 2j   -> index 0*p + 2j
    # f: row 0, col 2j+1 -> index 0*p + 2j+1
    # g: row 1, col 2j   -> index 1*p + 2j
    # h: row 1, col 2j+1 -> index 1*p + 2j+1
    
    # Indices for C block j (W)
    # C entries in W are at indices (convention W[k*n+i, r] where C[i,k] is row i, col k of C? 
    # Wait, the prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # Usually C is p x n.
    # If C is 2x8, rows are 0,1. Cols are 0..7.
    # Index in W: k*n + i.
    # If C[row, col], then k=row, i=col? Or k=col, i=row?
    # Standard trace convention: $C = A B$.
    # $C_{i,j} = \sum_k A_{i,k} B_{k,j}$.
    # The tensor is $T_{a,b,c}$.
    # $a$ indexes A, $b$ indexes B, $c$ indexes C.
    # The factor W has shape (p*n, rank).
    # If the convention is $W[k*n+i, r]$, and $C$ has dimensions $p \times n$ (rows $p$, cols $n$)?
    # No, $A$ is $n \times m$, $B$ is $m \times p$, $C$ is $n \times p$.
    # Wait, the prompt says "multiplying a 2 by 2 matrix by a 2 by 8 matrix".
    # A: 2x2. B: 2x8.
    # Result C: 2x8.
    # So n=2, m=2, p=8.
    # C is n x p = 2 x 8.
    # The prompt says "W[k*n+i,r]".
    # If C is 2x8, indices are row $i \in \{0,1\}$, col $j \in \{0..7\}$.
    # Usually flat index is $i \times p + j$.
    # But the prompt says $k \times n + i$.
    # If $n=2$, this is $k \times 2 + i$.
    # This looks like column-major indexing if $k$ is the column and $i$ is the row?
    # Or if $C$ is stored as $p \times n$?
    # Let's check the current program's verification.
    # `C_computed = C_computed.reshape(p, n).T`
    # This implies the flat array is in row-major order of a $p \times n$ matrix, which is then transposed to $n \times p$.
    # So the flat index $idx$ in the array of size $p \times n$ corresponds to $C_{i, j}$ where $j = idx // n$ and $i = idx \% n$?
    # If we reshape to $(p, n)$, element $(r, c)$ is at $r \times n + c$.
    # Then `.T` makes it $(n, p)$.
    # So $C_{i, j}$ in the final $(n, p)$ matrix corresponds to $C_{j, i}$ in the $(p, n)$ matrix.
    # So the flat index for $C_{i, j}$ (row $i$, col $j$ of the $2 \times 8$ result) is $j \times n + i$.
    # Here $j$ is the column index of C (0..7), $i$ is the row index of C (0..1).
    # So $W$ index is $col \times n + row$.
    
    # Let's map Strassen outputs to W indices.
    # For block $b$ (columns $2b, 2b+1$ of C):
    # C[0, 2b]   -> col 2b, row 0 -> idx (2b)*n + 0
    # C[0, 2b+1] -> col 2b+1, row 0 -> idx (2b+1)*n + 0
    # C[1, 2b]   -> col 2b, row 1 -> idx (2b)*n + 1
    # C[1, 2b+1] -> col 2b+1, row 1 -> idx (2b+1)*n + 1
    
    for b in range(num_blocks):
        base = b * 7
        # B block indices in V (flat index in 2x8 matrix, row-major: row*p + col)
        # e: B[0, 2b]   -> 0*p + 2b
        # f: B[0, 2b+1] -> 0*p + 2b+1
        # g: B[1, 2b]   -> 1*p + 2b
        # h: B[1, 2b+1] -> 1*p + 2b+1
        
        ve = 0 * p + 2 * b
        vf = 0 * p + 2 * b + 1
        vg = 1 * p + 2 * b
        vh = 1 * p + 2 * b + 1
        
        # W indices for C block b
        w00 = (2 * b) * n + 0     # C[0, 2b]
        w01 = (2 * b + 1) * n + 0 # C[0, 2b+1]
        w10 = (2 * b) * n + 1     # C[1, 2b]
        w11 = (2 * b + 1) * n + 1 # C[1, 2b+1]
        
        # M1 = (a + d)(e + h)
        # U: a(0), d(3)
        # V: e(ve), h(vh)
        # W: C00(w00), C11(w11)
        U[0][base + 0] += 1
        U[3][base + 0] += 1
        V[ve][base + 0] += 1
        V[vh][base + 0] += 1
        W[w00][base + 0] += 1
        W[w11][base + 0] += 1
        
        # M2 = (c + d)e
        # U: c(2), d(3)
        # V: e(ve)
        # W: C10(w10)
        U[2][base + 1] += 1
        U[3][base + 1] += 1
        V[ve][base + 1] += 1
        W[w10][base + 1] += 1
        
        # M3 = a(f - h)
        # U: a(0)
        # V: f(vf), h(vh)
        # W: C01(w01), C11(w11)
        U[0][base + 2] += 1
        V[vf][base + 2] += 1
        V[vh][base + 2] -= 1
        W[w01][base + 2] += 1
        W[w11][base + 2] += 1
        
        # M4 = d(g - e)
        # U: d(3)
        # V: g(vg), e(ve)
        # W: C00(w00), C10(w10)
        U[3][base + 3] += 1
        V[vg][base + 3] += 1
        V[ve][base + 3] -= 1
        W[w00][base + 3] += 1
        W[w10][base + 3] += 1
        
        # M5 = (a + b)h
        # U: a(0), b(1)
        # V: h(vh)
        # W: C00(w00) (-1), C10(w10) (+1)
        # Wait, Strassen formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's re-verify M5 contribution.
        # C00 has -M5. C01 has +M5.
        # So W[w00] -= 1, W[w01] += 1.
        # My previous code had:
        # W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        # W[(2 * b + 1) * n + 0][base + 4] += 1 # C[1,2b] -> This was C10?
        # Let's check indices.
        # w00 is C[0, 2b]. w01 is C[0, 2b+1]. w10 is C[1, 2b]. w11 is C[1, 2b+1].
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M5:
        # W[w00] -= 1
        # W[w01] += 1
        # W[w10] += 0
        # W[w11] += 0
        
        U[0][base + 4] += 1
        U[1][base + 4] += 1
        V[vh][base + 4] += 1
        W[w00][base + 4] -= 1
        W[w01][base + 4] += 1
        
        # M6 = (c - a)(e + f)
        # U: c(2), a(0)
        # V: e(ve), f(vf)
        # W: C11(w11)
        U[2][base + 5] += 1
        U[0][base + 5] -= 1
        V[ve][base + 5] += 1
        V[vf][base + 5] += 1
        W[w11][base + 5] += 1
        
        # M7 = (b - d)(g + h)
        # U: b(1), d(3)
        # V: g(vg), h(vh)
        # W: C00(w00)
        U[1][base + 6] += 1
        U[3][base + 6] -= 1
        V[vg][base + 6] += 1
        V[vh][base + 6] += 1
        W[w00][base + 6] += 1

    return {"U": U, "V": V, "W": W}
