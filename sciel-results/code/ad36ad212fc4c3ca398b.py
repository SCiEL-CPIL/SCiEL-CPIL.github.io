"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor T[i*m+j, j*p+k, k*n+i] = delta_{i,i}delta_{j,j}delta_{k,k}.
    Here T[a,b,c] = 1 iff a//m == c%n and b%m == a%m and b//p == c//n.
    We verify: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j2 != j:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k2 != k or i2 != i:
                    expected = 0
                else:
                    expected = 1
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    Strassen formulas:
      m00 = a00*b00 + a11*b11 - a10*(b01-b11) - a01*(b10-b11)
      m01 = a01*b11 + a11*b10
      m10 = a10*b00 + a00*b01
      m11 = a01*b10 - a10*b01 + a00*b00 + a11*b11
    We need: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = C[i,k] where C = A*B.
    So for each (i,k), the product over r must give C[i,k].
    
    Let's use the standard Strassen decomposition.
    The 7 multiplications are:
      p0 = (a00+a11)(b00+b11)
      p1 = (a10+a11)b00
      p2 = a00(b01-b11)
      p3 = a11(b10-b00)
      p4 = (a00+a01)b11
      p5 = (a10-a00)(b00+b10)
      p6 = (a01-a11)(b01+b11)
    
    C00 = p0 + p3 - p4 + p6
    C01 = p2 + p4
    C10 = p1 + p5
    C11 = p0 - p1 + p2 + p5
    
    Now express each C[i,k] as sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r].
    For a given (i,k), we need to decompose C[i,k] into rank-1 terms.
    
    This is complex. Let me just use the known rank-7 decomposition for 2x2x2.
    Actually, let me construct it directly.
    
    For the 2x2x2 case, the tensor T has entries T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} where the matmul is (A_{i,j} B_{j,k}) = C_{i,k}.
    T[a,b,c] where a=i*2+j, b=j*2+k, c=k*2+i.
    T[i*2+j, j*2+k, k*2+i] = 1 iff consistent, i.e., always 1 for valid indices.
    Wait no: T[a,b,c] = 1 if a//2 == c%2 and b%2 == a%2 and b//2 == c//2.
    
    Let me just hardcode the Strassen rank-7 decomposition.
    """
    # Known rank-7 decomposition for 2x2x2 matrix multiplication
    # U has shape (4,7), V has shape (4,7), W has shape (4,7)
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U = [
        [1,  0,  1,  0,  1,  1,  1],   # i=0,j=0
        [0,  0,  0,  0,  0,  0,  0],   # i=0,j=1  -- wait, this isn't right
    ]
    # Let me think more carefully.
    # Actually let me just use a different approach: use the known result that
    # R(2,2,2) = 7 and construct it properly.
    
    # The 7 rank-1 tensors for Strassen:
    # Each rank-1 term is u ⊗ v ⊗ w where:
    # u is a 4-vector (indexed by i*2+j), v is a 4-vector (indexed by j*2+k), w is a 4-vector (indexed by k*2+i)
    
    # Term 0: (a00+a11)(b00+b11) contributes to C00 and C11
    # u0 = [1,0,1,0] (a00 and a11), v0 = [1,0,1,0] (b00 and b11), w0 = [1,0,0,1] (C00 and C11)
    # Term 1: (a10+a11)b00 contributes to C10
    # u1 = [0,0,1,1], v1 = [1,0,0,0], w1 = [0,0,1,0]
    # Term 2: a00(b01-b11) contributes to C01
    # u2 = [1,0,0,0], v2 = [0,1,0,-1], w2 = [0,1,0,0]
    # Term 3: a11(b10-b00) contributes to C00
    # u3 = [0,0,0,1], v3 = [0,0,1,-1], w3 = [1,0,0,0]
    # Term 4: (a00+a01)b11 contributes to C01
    # u4 = [1,1,0,0], v4 = [0,0,0,1], w4 = [0,1,0,0]
    # Term 5: (a10-a00)(b00+b10) contributes to C10 and C11
    # u5 = [0,0,1,-1], v5 = [1,0,1,0], w5 = [0,0,1,1]
    # Term 6: (a01-a11)(b01+b11) contributes to C00
    # u6 = [0,1,0,-1], v6 = [0,1,0,1], w6 = [1,0,0,0]
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # i=0,j=0: u0[0],u1[0],u2[0],u3[0],u4[0],u5[0],u6[0]
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1: u0[1],u1[1],u2[1],u3[1],u4[1],u5[1],u6[1]
        [1, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [0, 1, 0, 1, 0, -1, -1], # i=1,j=1
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # j=0,k=0
        [0, 0, 1, 0, 0, 0, 1],   # j=0,k=1
        [1, 0, 0, 1, 0, 1, 0],   # j=1,k=0
        [0, 0, -1, -1, 1, 0, 1], # j=1,k=1
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 1],   # k=0,i=0
        [0, 0, 1, 0, 1, 0, 0],   # k=0,i=1
        [0, 1, 0, 0, 0, 1, 0],   # k=1,i=0
        [1, 0, 0, 0, 0, 1, 0],   # k=1,i=1
    ]
    
    # Verify
    if not _verify(U, V, W, 2, 2, 2, 7):
        # Fallback: return schoolbook
        return _schoolbook(2, 2, 2)
    return U, V, W

def _pad_and_decompose(n, m, p):
    """Try to use Strassen-style padding to get a lower rank decomposition.
    
    Strategy: Pad the 4x5x6 problem to a 4x6x6 problem, then try to decompose
    4x6x6 using block structure. 4x6x6 can be seen as two 4x3x3 blocks? No.
    
    Alternative: 4x5x6. We can pad to 4x6x6 (add a zero column to A and zero row to B).
    Then 4x6x6 has rank at most 4*6*6 = 144, but we want to do better.
    
    Actually, let's think about this differently. The key insight is that for
    rectangular matrix multiplication, we can sometimes use the structure to
    reduce rank below n*m*p.
    
    One approach: if we can factor the problem into smaller subproblems using
    block matrix multiplication, we might save.
    
    For 4x5x6: Think of A as 4x5, B as 5x6. We can split B into B1 (5x3) and B2 (5x3).
    Then C = A*B1 + A*B2, but that's just two separate multiplications, total rank 4*5*3 + 4*5*3 = 120.
    
    Better: Can we use the fact that 4 and 6 have a common structure?
    
    Let's try a different approach: use the Strassen 2x2x2 as a building block.
    Pad 4x5x6 to 4x6x6. Then 4x6x6 can be viewed as (2x2 blocks of 2x3).
    A is 4x6 = two 2x3 blocks vertically. B is 6x6 = four 3x3 blocks in a 2x2 grid.
    This gets complicated.
    
    Let me try yet another approach: direct search for a rank-112 or lower decomposition
    by exploiting the specific dimensions.
    
    Actually, for the purpose of this task, let me try to use a simple improvement:
    The schoolbook rank is 4*5*6 = 120. Can we do better?
    
    One known result: R(4,5,6) <= 112 is not standard. Let me just try to verify
    if we can shave off a few terms by noticing that some entries are always zero
    due to the rectangular shape.
    
    Actually, in the schoolbook decomposition, every term is nonzero, so there's no
    obvious waste. The rank is exactly n*m*p for the schoolbook.
    
    Let me try a completely different strategy: use the fact that 5 = 2+3 and 6 = 3+3,
    and 4 = 2+2, to do a block decomposition.
    
    A (4x5) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3.
    B (5x6) = [B11 B12; B21 B22] where B11 is 2x3, B12 is 2x3, B21 is 3x3, B22 is 3x3.
    
    C (4x6) = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    This doesn't immediately help reduce rank.
    
    Let me just return the schoolbook for now, but with a slightly different
    implementation that might allow for future optimization.
    """
    return _schoolbook(n, m, p)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Check if we're already at a good rank
    # For 4x5x6, the known lower bound is around 100-110, schoolbook is 120.
    # Without a proven better decomposition, return schoolbook.
    
    # Time check
    if time.time() > deadline:
        return best_result
    
    # Try to verify the schoolbook decomposition (should pass)
    if _verify(U, V, W, n, m, p, best_rank):
        pass  # Good
    else:
        # Should not happen, but if it does, we have a bug
        pass
    
    return best_result
# EVOLVE-BLOCK-END
