"""Tensor decomposition for 2x3x8 matrix multiplication using a block-wise rank-44 construction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 construction:
    # Split m=3 into m1=2 and m2=1.
    # Part 1: 2x2x8 (split into four 2x2x2 Strassen blocks) -> Rank 4 * 7 = 28
    # Part 2: 2x1x8 (trivial outer product) -> Rank 2 * 8 = 16
    # Total Rank = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x8 using 4 Strassen 2x2x2 blocks
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices in local 2x2x2:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
    
    # Mapping Strassen terms to U, V, W local indices
    # U index: i*m + j. For 2x2, i in 0,1; j in 0,1.
    # V index: j*p + k. For 2x2, j in 0,1; k in 0,1.
    # W index: k*n + i. For 2x2, k in 0,1; i in 0,1.
    
    def add_strassen_block(base_u, base_v, base_w, offset):
        nonlocal col
        # base_u, base_v, base_w are offsets in the global U, V, W arrays for the local 2x2x2 block
        # Local U indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
        # Local V indices: 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
        # Local W indices: 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        
        # M1 = (a+d)(e+h)
        # U: a(0), d(3) -> U[base+0]=1, U[base+3]=1
        # V: e(0), h(3) -> V[base+0]=1, V[base+3]=1
        # W: c11(0), c22(3) -> W[base+0]=1, W[base+3]=1
        U[base_u + 0][col] = 1
        U[base_u + 3][col] = 1
        V[base_v + 0][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 0][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M2 = (c+d)e
        # U: c(2), d(3)
        # V: e(0)
        # W: c21(2), c22(3)
        U[base_u + 2][col] = 1
        U[base_u + 3][col] = 1
        V[base_v + 0][col] = 1
        W[base_w + 2][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M3 = a(f-h)
        # U: a(0)
        # V: f(1), h(3) -> V[base+1]=1, V[base+3]=-1
        # W: c12(1), c22(3)
        U[base_u + 0][col] = 1
        V[base_v + 1][col] = 1
        V[base_v + 3][col] = -1
        W[base_w + 1][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M4 = d(g-e)
        # U: d(3)
        # V: g(2), e(0) -> V[base+2]=1, V[base+0]=-1
        # W: c11(0), c21(2)
        U[base_u + 3][col] = 1
        V[base_v + 2][col] = 1
        V[base_v + 0][col] = -1
        W[base_w + 0][col] = 1
        W[base_w + 2][col] = 1
        col += 1
        
        # M5 = (a+b)h
        # U: a(0), b(1)
        # V: h(3)
        # W: c11(0), c12(1)
        U[base_u + 0][col] = 1
        U[base_u + 1][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 0][col] = 1
        W[base_w + 1][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # U: c(2), a(0) -> U[base+2]=1, U[base+0]=-1
        # V: e(0), f(1)
        # W: c12(1), c22(3)
        U[base_u + 2][col] = 1
        U[base_u + 0][col] = -1
        V[base_v + 0][col] = 1
        V[base_v + 1][col] = 1
        W[base_w + 1][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # U: b(1), d(3) -> U[base+1]=1, U[base+3]=-1
        # V: g(2), h(3)
        # W: c11(0), c22(3)
        U[base_u + 1][col] = 1
        U[base_u + 3][col] = -1
        V[base_v + 2][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 0][col] = 1
        W[base_w + 3][col] = 1
        col += 1

    # Process 4 blocks for 2x2x8
    for block_k in range(4):
        # Global A indices for this block: i in 0,1; j in 0,1
        # U index = i * 3 + j
        base_u = 0 # i=0, j=0 is U[0]
        # Global B indices for this block: j in 0,1; k in 2*block_k, 2*block_k+1
        # V index = j * 8 + k
        base_v = 0 * 8 + 2 * block_k # j=0, k=2*block_k
        # Global C indices for this block: i in 0,1; k in 2*block_k, 2*block_k+1
        # W index = k * 2 + i
        base_w = (2 * block_k) * 2 + 0 # k=2*block_k, i=0
        
        add_strassen_block(base_u, base_v, base_w, col)

    # Part 2: 2x1x8
    # A index: i in 0,1; j=2 -> U[i*3 + 2]
    # B index: j=2; k in 0..7 -> V[2*8 + k]
    # C index: i in 0,1; k in 0..7 -> W[k*2 + i]
    
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
