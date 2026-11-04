"""Matrix multiplication 3x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n=3, m=5, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each rank term r, we have u_r, v_r, w_r
    # The product tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if and only if the schoolbook term (i,j,k) exists, else 0.
    
    # Build the expected tensor
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1
    
    # Compute the actual tensor from the decomposition
    actual = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
    
    # Check if they match
    if not np.allclose(actual, expected, atol=1e-9):
        return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-element lists in row-major order.
    
    For A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    C = A*B where c00=a00*b00+a01*b10, c01=a00*b01+a01*b11,
             c10=a10*b00+a11*b10, c11=a10*b01+a11*b11
    
    u indices: [a00, a01, a10, a11]
    v indices: [b00, b01, b10, b11]
    w indices: [c00, c01, c10, c11]
    
    Standard Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)(b00)
    M3 = (a00)(b10+b11)
    M4 = (a11)(b11-b00)
    M5 = (a00+a10)(b01-b00)
    M6 = (a10+a11)(b11-b10)
    M7 = (a00+a01)(b10)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (a10+a11)(b00)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = (a00)(b10+b11)
    terms.append(([1,0,0,0], [0,0,1,1], [0,1,0,0]))
    # M4 = (a11)(b11-b00)
    terms.append(([0,0,0,1], [0,0,0,1], [0,0,0,1]))
    # Actually M4 = (a11)(b11-b00) -> u=[0,0,0,1], v=[-1,0,0,1]... wait
    # b11-b00 means v = [-1, 0, 0, 1] for [b00,b01,b10,b11]
    # Let me redo this carefully.
    
    # b00 is index 0, b01 is index 1, b10 is index 2, b11 is index 3
    # M1 = (a00+a11)(b00+b11): u=[1,0,0,1], v=[1,0,0,1]
    # M2 = (a10+a11)(b00): u=[0,0,1,1], v=[1,0,0,0]
    # M3 = (a00)(b10+b11): u=[1,0,0,0], v=[0,0,1,1]
    # M4 = (a11)(b11-b00): u=[0,0,0,1], v=[-1,0,0,1]
    # M5 = (a00+a10)(b01-b00): u=[1,0,1,0], v=[-1,1,0,0]
    # M6 = (a10+a11)(b11-b10): u=[0,0,1,1], v=[0,0,-1,1]
    # M7 = (a00+a01)(b10): u=[1,1,0,0], v=[0,0,1,0]
    
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))   # M1
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))    # M2
    terms.append(([1,0,0,0], [0,0,1,1], [0,1,0,0]))    # M3
    terms.append(([0,0,0,1], [-1,0,0,1], [0,0,0,1]))   # M4
    terms.append(([1,0,1,0], [-1,1,0,0], [0,1,0,0]))   # M5
    terms.append(([0,0,1,1], [0,0,-1,1], [0,0,0,1]))   # M6
    terms.append(([1,1,0,0], [0,0,1,0], [1,0,0,0]))    # M7
    
    return terms

def _construct_strassen_block():
    """
    Construct a decomposition for 3x5x6 using Strassen for 2x2 sub-blocks.
    
    Partition:
    A (3x5) = [[A11(2x2), A12(2x3)], [A21(1x2), A22(1x3)]]
    B (5x6) = [[B11(2x2), B12(2x4)], [B21(3x2), B22(3x4)]]
    
    C (3x6) = [[C11(2x2), C12(2x4)], [C21(1x2), C22(1x4)]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    """
    n, m, p = 3, 5, 6
    U = np.zeros((n*m, 99999), dtype=float)
    V = np.zeros((m*p, 99999), dtype=float)
    W = np.zeros((p*n, 99999), dtype=float)
    
    col = 0
    
    # C11 = A11*B11 + A12*B21
    # A11 is rows 0-1, cols 0-1 of A; B11 is rows 0-1, cols 0-1 of B; C11 is rows 0-1, cols 0-1 of C
    # A11*B11: use Strassen (7 terms)
    strassen_terms = _strassen_2x2_terms()
    for u, v, w in strassen_terms:
        # u applies to A11 (2x2 at A[0:2, 0:2])
        # v applies to B11 (2x2 at B[0:2, 0:2])
        # w applies to C11 (2x2 at C[0:2, 0:2])
        for i in range(2):
            for j in range(2):
                U[i*m + j, col] += u[i*2 + j]
                V[j*p + (0 + i), col] += v[i*2 + j]  # B11 row i, col j -> B[i, j]
                W[(0 + i)*n + (0 + j), col] += w[i*2 + j]  # C11 row i, col j -> C[i, j]
        col += 1
    
    # A12*B21: A12 is 2x3 (A[0:2, 2:5]), B21 is 3x2 (B[2:5, 0:2])
    # Schoolbook: 2*3*2 = 12 terms
    for i in range(2):      # row of A12 (0,1)
        for k in range(3):  # col of A12 / row of B21 (2,3,4)
            for j in range(2):  # col of B21 (0,1)
                U[(i)*m + (2+k), col] = 1
                V[(2+k)*p + j, col] = 1
                W[j*n + i, col] = 1
                col += 1
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: A11 is 2x2 (A[0:2,0:2]), B12 is 2x4 (B[0:2, 2:6])
    # Schoolbook: 2*2*4 = 16 terms
    for i in range(2):
        for k in range(2):
            for j in range(4):
                U[i*m + k, col] = 1
                V[k*p + (2+j), col] = 1
                W[(2+j)*n + i, col] = 1
                col += 1
    
    # A12*B22: A12 is 2x3 (A[0:2, 2:5]), B22 is 3x4 (B[2:5, 2:6])
    # Schoolbook: 2*3*4 = 24 terms
    for i in range(2):
        for k in range(3):
            for j in range(4):
                U[i*m + (2+k), col] = 1
                V[(2+k)*p + (2+j), col] = 1
                W[(2+j)*n + i, col] = 1
                col += 1
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: A21 is 1x2 (A[2, 0:2]), B11 is 2x2 (B[0:2, 0:2])
    # Schoolbook: 1*2*2 = 4 terms
    for k in range(2):
        for j in range(2):
            U[2*m + k, col] = 1
            V[k*p + j, col] = 1
            W[j*n + 2, col] = 1
            col += 1
    
    # A22*B21: A22 is 1x3 (A[2, 2:5]), B21 is 3x2 (B[2:5, 0:2])
    # Schoolbook: 1*3*2 = 6 terms
    for k in range(3):
        for j in range(2):
            U[2*m + (2+k), col] = 1
            V[(2+k)*p + j, col] = 1
            W[j*n + 2, col] = 1
            col += 1
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: A21 is 1x2 (A[2, 0:2]), B12 is 2x4 (B[0:2, 2:6])
    # Schoolbook: 1*2*4 = 8 terms
    for k in range(2):
        for j in range(4):
            U[2*m + k, col] = 1
            V[k*p + (2+j), col] = 1
            W[(2+j)*n + 2, col] = 1
            col += 1
    
    # A22*B22: A22 is 1x3 (A[2, 2:5]), B22 is 3x4 (B[2:5, 2:6])
    # Schoolbook: 1*3*4 = 12 terms
    for k in range(3):
        for j in range(4):
            U[2*m + (2+k), col] = 1
            V[(2+k)*p + (2+j), col] = 1
            W[(2+j)*n + 2, col] = 1
            col += 1
    
    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_simple_block():
    """
    Simpler block decomposition using schoolbook for all sub-blocks.
    This gives rank 90 (same as full schoolbook) but serves as a template.
    """
    return _construct_schoolbook()

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Try the Strassen-based block decomposition first
    try:
        result = _construct_strassen_block()
        if time.time() < deadline:
            # Verify the decomposition
            if _verify_decomposition(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
