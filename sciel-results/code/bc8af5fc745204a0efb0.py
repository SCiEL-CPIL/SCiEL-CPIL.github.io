"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the multiplication tensor has all-ones entries)
    
    # Check a few random (i,j,k) triples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        if abs(val - 1) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100 second deadline
    
    # Try 6 disjoint 2x2x2 Strassen blocks
    # n=3: Strassen on {0,1}, schoolbook on {2}
    # m=5: Strassen on {0,1} and {2,3}, schoolbook on {4}
    # p=7: Strassen on {0,1}, {2,3}, {4,5}, schoolbook on {6}
    
    # Strassen blocks: n_blocks=1, m_blocks=2, p_blocks=3 → 6 blocks
    # Each block: 7 terms
    # Remaining: 1*1*1 = 1 schoolbook term
    # Total rank: 6*7 + 1 = 43? No, that's not right.
    
    # Actually, the remaining terms are:
    # For n=2 (index 2): all j,k → 5*7 = 35 terms
    # For n={0,1}, m=4 (index 4): all k → 2*7 = 14 terms
    # For n={0,1}, m={0,1,2,3}, p=6 (index 6): 2*4*1 = 8 terms
    # For n={0,1}, m=4, p={0,1,2,3,4,5}: 2*1*6 = 12 terms
    # Wait, this is getting complicated. Let me just count.
    
    # Total triples: 3*5*7 = 105
    # Strassen blocks cover: 6*8 = 48 triples
    # Remaining: 105 - 48 = 57 triples, each needing 1 term
    # Total rank: 6*7 + 57 = 42 + 57 = 99
    
    # Let me implement this.
    
    U = {}
    V = {}
    W = {}
    
    def add_term(i, j, k, u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term."""
        # u_coeff is a dict mapping (i,j) to coefficient
        # v_coeff is a dict mapping (j,k) to coefficient
        # w_coeff is a dict mapping (k,i) to coefficient
        pass
    
    # This is getting too complicated. Let me just implement it directly.
    
    # Strassen 2x2x2 decomposition:
    # For block with n-indices {a,b}, m-indices {c,d}, p-indices {e,f}:
    # Using standard Strassen:
    # M1 = (A[a,c]+A[b,d]) * (B[c,e]+B[d,f])
    # M2 = (A[b,c]+A[b,d]) * B[c,e]
    # M3 = A[a,c] * (B[d,f]-B[d,e])
    # M4 = A[b,d] * (B[c,f]-B[c,e])
    # M5 = (A[a,c]+A[a,d]) * B[d,f]
    # M6 = (A[b,c]-A[a,c]) * (B[c,e]+B[c,f])
    # M7 = (A[b,d]-A[a,d]) * (B[d,e]+B[d,f])
    
    # C[e,a] = M1 + M4 - M5 - M7
    # C[e,f] = M3 + M5
    # C[f,a] = M2 + M6
    # C[f,b] = M1 - M2 + M3 + M7
    
    # Let me verify this numerically.
    
    a, b = 0, 1  # n-indices
    c, d = 0, 1  # m-indices
    e, f = 0, 1  # p-indices
    
    # Test with specific values
    A = np.zeros((2,2))
    B = np.zeros((2,2))
    A[0,0], A[0,1], A[1,0], A[1,1] = 1, 2, 3, 4
    B[0,0], B[0,1], B[1,0], B[1,1] = 5, 6, 7, 8
    
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[1,1]-B[1,0])
    M4 = A[1,1] * (B[0,1]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M6
    C11 = M1 - M2 + M3 + M7
    
    expected = A @ B
    
    print(f"Strassen test: C00={C00} (expected {expected[0,0]}), C01={C01} (expected {expected[0,1]}), C10={C10} (expected {expected[1,0]}), C11={C11} (expected {expected[1,1]})")
    
    # If this doesn't work, fall back to schoolbook
    if not (abs(C00-expected[0,0])<1e-9 and abs(C01-expected[0,1])<1e-9 and abs(C10-expected[1,0])<1e-9 and abs(C11-expected[1,1])<1e-9):
        # Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen works! Now implement the full 6-block decomposition.
    
    # This is getting too long for the thinking. Let me just write the complete code.
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    n_blocks = [(0, 1)]
    m_blocks = [(0, 1), (2, 3)]
    p_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for (a, b) in n_blocks:
        for (c, d) in m_blocks:
            for (e, f) in p_blocks:
                # M1 = (A[a,c]+A[b,d]) * (B[c,e]+B[d,f])
                U[a*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] += 1  # C[e,a]
                W[f*n+b][col] += 1  # C[f,b]
                # Wait, C[e,a] = M1 + M4 - M5 - M7, so M1 contributes +1 to C[e,a]
                # C[f,b] = M1 - M2 + M3 + M7, so M1 contributes +1 to C[f,b]
                # But C[e,f] and C[f,a] don't get M1.
                # So W for M1: W[e*n+a] += 1, W[f*n+b] += 1
                col += 1
                
                # M2 = (A[b,c]+A[b,d]) * B[c,e]
                U[b*m+c][col] += 1
                U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                W[f*n+a][col] += 1  # C[f,a] = M2 + M6
                col += 1
                
                # M3 = A[a,c] * (B[d,f]-B[d,e])
                U[a*m+c][col] += 1
                V[d*p+f][col] += 1
                V[d*p+e][col] -= 1
                W[e*n+f][col] += 1  # C[e,f] = M3 + M5
                col += 1
                
                # M4 = A[b,d] * (B[c,f]-B[c,e])
                U[b*m+d][col] += 1
                V[c*p+f][col] += 1
                V[c*p+e][col] -= 1
                W[e*n+a][col] += 1  # C[e,a] = M1 + M4 - M5 - M7
                col += 1
                
                # M5 = (A[a,c]+A[a,d]) * B[d,f]
                U[a*m+c][col] += 1
                U[a*m+d][col] += 1
                V[d*p+f][col] += 1
                W[e*n+f][col] += 1  # C[e,f] = M3 + M5
                W[e*n+a][col] -= 1  # C[e,a] = M1 + M4 - M5 - M7
                col += 1
                
                # M6 = (A[b,c]-A[a,c]) * (B[c,e]+B[c,f])
                U[b*m+c][col] += 1
                U[a*m+c][col] -= 1
                V[c*p+e][col] += 1
                V[c*p+f][col] += 1
                W[f*n+a][col] += 1  # C[f,a] = M2 + M6
                col += 1
                
                # M7 = (A[b,d]-A[a,d]) * (B[d,e]+B[d,f])
                U[b*m+d][col] += 1
                U[a*m+d][col] -= 1
                V[d*p+e][col] += 1
                V[d*p+f][col] += 1
                W[e*n+a][col] -= 1  # C[e,a] = M1 + M4 - M5 - M7
                W[f*n+b][col] += 1  # C[f,b] = M1 - M2 + M3 + M7
                col += 1
    
    # Schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this triple is covered by a Strassen block
                covered = False
                for (a, b) in n_blocks:
                    if i in (a, b):
                        for (c, d) in m_blocks:
                            if j in (c, d):
                                for (e, f) in p_blocks:
                                    if k in (e, f):
                                        covered = True
                if not covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    import random
    random.seed(123)
    for _ in range(50):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
        if abs(val - 1) > 1e-9:
            print(f"Verification failed at i={i}, j={j}, k={k}: val={val}")
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
