"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _laderman_3x3x3():
    """
    Return the Laderman rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 23) where:
      U[i*3+j, t] = coefficient of A[i,j] in the t-th scalar product
      V[j*3+k, t] = coefficient of B[j,k] in the t-th scalar product
      W[k*3+i, t] = coefficient of the t-th scalar product in C[i,k]
    such that C[i,k] = sum_j sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t]
    """
    # Laderman's 23-term decomposition for 3x3x3.
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # Products P1..P23 and reconstruction of C = A*B.
    
    # I will encode the 23 terms as (alpha, beta, gamma) vectors of length 9 each.
    # alpha: coefficients for A entries [a,b,c,d,e,f,g,h,i]
    # beta:  coefficients for B entries [j,k,l,m,n,o,p,q,r]
    # gamma: coefficients for C entries [C00,C01,C02,C10,C11,C12,C20,C21,C22]
    
    terms = [
        # P1 = a * (j - m + p + q)
        ([1,0,0, 0,0,0, 0,0,0],
         [1,0,0, -1,0,0, 1,1,0],
         [1,0,0, 0,0,0, 0,0,1]),
        # P2 = (a+b) * m
        ([1,1,0, 0,0,0, 0,0,0],
         [0,0,0, 1,0,0, 0,0,0],
         [0,0,1, 0,0,0, 0,0,0]),
        # P3 = (a+c) * p
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,0],
         [0,0,1, 0,0,0, 0,0,0]),
        # P4 = i * (j + k - m - n)
        ([0,0,0, 0,0,0, 0,0,1],
         [1,1,0, -1,-1,0, 0,0,0],
         [1,0,0, 0,0,0, 0,0,1]),
        # P5 = (g+i) * k
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,0, 0,0,0, 0,0,0],
         [0,1,0, 0,0,0, 0,0,0]),
        # P6 = (g+i) * n
        ([0,0,0, 0,0,0, 1,0,1],
         [0,0,0, 0,1,0, 0,0,0],
         [0,0,0, 0,1,0, 0,0,0]),
        # P7 = (c+i) * (m + n - p - q)
        ([0,0,1, 0,0,0, 0,0,1],
         [0,0,0, 1,1,0, -1,-1,0],
         [0,1,0, 0,0,0, 0,1,0]),
        # P8 = (a+c+g+i-e) * (j+r)
        ([1,0,1, 0,-1,0, 1,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [1,0,0, 0,0,0, 0,0,-1]),
        # P9 = (c+i-e) * (j+r)
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [0,0,0, 1,0,0, 0,0,0]),
        # P10 = (a+c-e) * (p+r)
        ([1,0,1, 0,-1,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,1],
         [1,0,0, 0,0,0, 1,0,0]),
        # P11 = (a+c) * (p+r-m-o)
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, -1,0,-1, 1,0,1],
         [0,1,0, 0,0,0, 0,0,0]),
        # P12 = (g+i) * (k+l-n-o)
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,1, 0,-1,-1, 0,0,0],
         [0,0,0, 0,0,1, 0,0,0]),
        # P13 = (g+i-e) * (k+l)
        ([0,0,0, 0,-1,0, 1,0,1],
         [0,1,1, 0,0,0, 0,0,0],
         [0,0,1, 0,0,0, 0,0,0]),
        # P14 = (c+i-e) * (j+l)
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,1, 0,0,0, 0,0,0],
         [1,0,0, 0,0,0, 1,0,0]),
        # P15 = (c+i) * (j+l-m-o)
        ([0,0,1, 0,0,0, 0,0,1],
         [1,0,1, -1,0,-1, 0,0,0],
         [0,1,0, 0,0,0, 0,1,0]),
        # P16 = b * (n-q)
        ([0,1,0, 0,0,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,0,0, 0,1,0, 0,0,0]),
        # P17 = d * (j-m)
        ([0,0,0, 1,0,0, 0,0,0],
         [1,0,0, -1,0,0, 0,0,0],
         [0,0,0, 1,0,0, 0,0,0]),
        # P18 = e * (n-q)
        ([0,0,0, 0,1,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,0,0, 0,0,0, 0,0,0]),
        # P19 = f * (p-r)
        ([0,0,0, 0,0,1, 0,0,0],
         [0,0,0, 0,0,0, 1,0,-1],
         [0,0,0, 0,0,0, 0,0,0]),
        # P20 = h * (k-l)
        ([0,0,0, 0,0,0, 0,1,0],
         [0,1,-1, 0,0,0, 0,0,0],
         [0,0,0, 0,1,0, 0,0,0]),
        # P21 = h * (n-o)
        ([0,0,0, 0,0,0, 0,1,0],
         [0,0,0, 0,1,-1, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
        # P22 = d * (n+o)
        ([0,0,0, 1,0,0, 0,0,0],
         [0,0,0, 0,1,1, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
        # P23 = f * (j+l)
        ([0,0,0, 0,0,1, 0,0,0],
         [1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
    ]
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    for t, (alpha, beta, gamma) in enumerate(terms):
        U3[:, t] = np.array(alpha, dtype=float)
        V3[:, t] = np.array(beta, dtype=float)
        W3[:, t] = np.array(gamma, dtype=float)
    
    return U3, V3, W3


def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    for _ in range(num_tests):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for t in range(U.shape[1]):
                        s += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 145.0
    
    # Get Laderman 3x3x3 decomposition
    U3, V3, W3 = _laderman_3x3x3()
    
    # Verify Laderman for 3x3x3 first
    if not _verify_decomposition(U3, V3, W3, 3, 3, 3, num_tests=3):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 3x3x6 decomposition using two copies of Laderman.
    # C = A * B where B is 3x6. Split B = [B1 | B2] with B1, B2 each 3x3.
    # C1 = A * B1 (3x3), C2 = A * B2 (3x3)
    # C = [C1 | C2]
    
    rank = 46
    U = np.zeros((n*m, rank))   # (9, 46)
    V = np.zeros((m*p, rank))   # (18, 46)
    W = np.zeros((p*n, rank))   # (18, 46)
    
    # First block: columns 0-2 of B
    for t in range(23):
        # U: same for both blocks
        U[:, t] = U3[:, t]
        # V: rows 0-8 (j*3+k for j in 0..2, k in 0..2)
        V[0:9, t] = V3[:, t]
        # W: rows 0-8 (k*3+i for k in 0..2, i in 0..2)
        W[0:9, t] = W3[:, t]
    
    # Second block: columns 3-5 of B
    for t in range(23):
        t2 = t + 23
        # U: same
        U[:, t2] = U3[:, t]
        # V: rows 9-17 (j*6+(k+3) for j in 0..2, k in 0..2)
        for j in range(3):
            for k in range(3):
                V[j*6 + (k+3), t2] = V3[j*3 + k, t]
        # W: rows 9-17 ((k+3)*3+i for k in 0..2, i in 0..2)
        for k in range(3):
            for i in range(3):
                W[(k+3)*3 + i, t2] = W3[k*3 + i, t]
    
    # Verify the full 3x3x6 decomposition
    if _verify_decomposition(U, V, W, n, m, p, num_tests=3):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
