"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 3, 5, 7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Try Laderman-based decomposition with rank 17
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 2x3x3 Laderman (i in {0,1}, j in {0,1,2}, k in {0,1,2})
    # I'll try the following 7 factors:
    # Factor 0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    # Factor 1: U[1,0]=1, V[0,1]=1, W[1,1]=1
    # Factor 2: U[0,1]=1, V[1,0]=1, W[0,0]=1
    # Factor 3: U[1,1]=1, V[1,1]=1, W[1,1]=1
    # Factor 4: U[0,2]=1, V[2,0]=1, W[0,0]=1
    # Factor 5: U[1,2]=1, V[2,1]=1, W[1,1]=1
    # Factor 6: U[0,0]=1, U[1,0]=1, V[0,2]=1, W[0,2]=1, W[1,2]=1
    
    factors = [
        # (u_entries, v_entries, w_entries)
        ([(0,0)], [(0,0)], [(0,0)]),
        ([(1,0)], [(0,1)], [(1,1)]),
        ([(0,1)], [(1,0)], [(0,0)]),
        ([(1,1)], [(1,1)], [(1,1)]),
        ([(0,2)], [(2,0)], [(0,0)]),
        ([(1,2)], [(2,1)], [(1,1)]),
        ([(0,0),(1,0)], [(0,2)], [(0,2),(1,2)]),
    ]
    
    for r, (u_e, v_e, w_e) in enumerate(factors):
        for (i, j) in u_e:
            U[i * m + j][r] = 1
        for (j, k) in v_e:
            V[j * p + k][r] = 1
        for (k, i) in w_e:
            W[k * n + i][r] = 1
    
    # Block 2: 3x2x3 Laderman (i in {0,1,2}, j in {3,4}, k in {0,1,2})
    # This is a 3x2x3 tensor. I'll use a similar Laderman decomposition.
    # For 3x2x3, the tensor T[i,j,k] = 1 for i in {0,1,2}, j in {3,4}, k in {0,1,2}
    # I'll shift j by 3: j' = j - 3, so j' in {0,1}
    # This is like a 3x2x3 tensor. I'll use Laderman's algorithm for 3x2x3.
    
    # Laderman 3x2x3 factors (7 factors):
    # Factor 0: U[0,3]=1, V[3,0]=1, W[0,0]=1
    # Factor 1: U[0,4]=1, V[4,0]=1, W[0,0]=1
    # Factor 2: U[1,3]=1, V[3,1]=1, W[1,1]=1
    # Factor 3: U[1,4]=1, V[4,1]=1, W[1,1]=1
    # Factor 4: U[2,3]=1, V[3,2]=1, W[2,2]=1
    # Factor 5: U[2,4]=1, V[4,2]=1, W[2,2]=1
    # Factor 6: U[0,3], U[1,3], U[2,3], V[3,0], V[3,1], V[3,2], W[0,0], W[1,1], W[2,2]
    
    # Actually, I'm not sure this is correct. Let me just use schoolbook for block 2.
    # Block 2 has 3*2*3 = 18 entries, so schoolbook rank is 18.
    # Total rank would be 7 + 18 + 3 = 28.
    
    # Let me just use schoolbook for all blocks to ensure correctness.
    # Rank = 3*5*7 = 105.
    
    # Actually, let me verify the Laderman decomposition for block 1.
    # I'll check if the decomposition is correct.
    
    # For block 1 (i in {0,1}, j in {0,1,2}, k in {0,1,2}):
    # T[i,j,k] should be 1 for all i,j,k.
    # Let me compute the sum for each (i,j,k).
    
    # This is getting complex. Let me just return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
