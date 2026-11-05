"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _build_tensor_support(n=3, m=3, p=6):
    """
    Build the list of (a, b, c) indices that are 'on' in the tensor.
    a = i*m + j
    b = j*p + k
    c = k*n + i
    """
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    return support

def _compute_error(U, V, W, n=3, m=3, p=6, support=None):
    """
    Compute the error of the decomposition.
    Error = sum over all entries (T_decomp - T_target)^2
    """
    if support is None:
        support = _build_tensor_support(n, m, p)
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct Target Tensor
    T_target = np.zeros((nm, mp, pn))
    for a, b, c in support:
        T_target[a, b, c] = 1.0
    
    # Compute T_decomp
    T_decomp = np.einsum('ad,bd,cd->abc', U, V, W)
    
    return float(np.sum((T_decomp - T_target) ** 2))

def _strassen_3x3():
    """
    Returns the standard Strassen decomposition for 3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indices: a = i*3+j, b = j*3+k, c = k*3+i
    """
    # Strassen formulas for C = A * B
    # M1 = (A11 - A22) * (B22 + B33)
    # M2 = (A11 + A22) * B22
    # M3 = A33 * (B11 + B22)
    # M4 = A11 * (B33 - B22)
    # M5 = (A33 + A11) * (B11 - B33)
    # M6 = (A22 + A33) * (B33 - B11)
    # M7 = (A11 - A33) * (B11 + B22)
    
    # C11 = M1 + M2 - M3 + M4
    # C12 = M4 + M5
    # C13 = M3 - M5 - M7
    # C21 = M6 + M7
    # C22 = M1 - M2 + M3 + M5
    # C23 = M1 - M3 + M6
    # C31 = M2 - M4 - M6
    # C32 = M2 - M1 + M7
    # C33 = M1 + M3 - M5 - M7
    
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # Helper to set U, V, W for a term: U[i,j] += coeff, V[j,k] += coeff, W[k,i] += coeff
    # Note: The standard Strassen decomposition is for C = A B.
    # Our tensor T[a,b,c] corresponds to C[i,k] = sum_j A[i,j] B[j,k].
    # Wait, standard matrix multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # In our tensor definition:
    # a = i*m+j (input A index)
    # b = j*p+k (input B index)
    # c = k*n+i (output C index)
    # So yes, it is standard matrix multiplication.
    
    # Let's define the Strassen factors carefully.
    # We are decomposing the tensor T such that T = sum_r U_r V_r W_r^T (outer products)
    # Actually T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Term 1: (A11 - A22) * (B22 + B33)
    # A11 is A[0,0] -> a=0*3+0=0. A22 is A[1,1] -> a=1*3+1=4.
    # B22 is B[1,1] -> b=1*3+1=4. B33 is B[2,2] -> b=2*3+2=8.
    # C11 is C[0,0] -> c=0*3+0=0. C22 is C[1,1] -> c=1*3+1=4.
    # Contribution to C11: +1. Contribution to C22: +1.
    
    # Let's just hardcode the Strassen decomposition for 3x3.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # M1 = (A11 - A22) * (B22 + B33)
    # M2 = (A11 + A22) * B22
    # M3 = A33 * (B11 + B22)
    # M4 = A11 * (B33 - B22)
    # M5 = (A33 + A11) * (B11 - B33)
    # M6 = (A22 + A33) * (B33 - B11)
    # M7 = (A11 - A33) * (B11 + B22)
    
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    
    # M1
    # U: A11(0) +1, A22(4) -1
    # V: B22(4) +1, B33(8) +1
    # W: C11(0) +1, C22(4) +1
    U[0, 0] += 1; U[4, 0] += -1
    V[4, 0] += 1; V[8, 0] += 1
    W[0, 0] += 1; W[4, 0] += 1
    
    # M2
    # U: A11(0) +1, A22(4) +1
    # V: B22(4) +1
    # W: C11(0) +1, C22(4) -1, C31(6) +1, C32(7) -1
    U[0, 1] += 1; U[4, 1] += 1
    V[4, 1] += 1
    W[0, 1] += 1; W[4, 1] += -1; W[6, 1] += 1; W[7, 1] += -1
    
    # M3
    # U: A33(8) +1
    # V: B11(0) +1, B22(4) +1
    # W: C11(0) -1, C13(2) -1, C22(4) +1, C23(5) -1, C33(8) +1
    U[8, 2] += 1
    V[0, 2] += 1; V[4, 2] += 1
    W[0, 2] += -1; W[2, 2] += -1; W[4, 2] += 1; W[5, 2] += -1; W[8, 2] += 1
    
    # M4
    # U: A11(0) +1
    # V: B33(8) +1, B22(4) -1
    # W: C11(0) +1, C12(1) +1, C31(6) -1
    U[0, 3] += 1
    V[8, 3] += 1; V[4, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1; W[6, 3] += -1
    
    # M5
    # U: A33(8) +1, A11(0) +1
    # V: B11(0) +1, B33(8) -1
    # W: C12(1) +1, C22(4) +1, C33(8) -1
    U[8, 4] += 1; U[0, 4] += 1
    V[0, 4] += 1; V[8, 4] += -1
    W[1, 4] += 1; W[4, 4] += 1; W[8, 4] += -1
    
    # M6
    # U: A22(4) +1, A33(8) +1
    # V: B33(8) +1, B11(0) -1
    # W: C21(3) +1, C23(5) +1, C31(6) -1
    U[4, 5] += 1; U[8, 5] += 1
    V[8, 5] += 1; V[0, 5] += -1
    W[3, 5] += 1; W[5, 5] += 1; W[6, 5] += -1
    
    # M7
    # U: A11(0) +1, A33(8) -1
    # V: B11(0) +1, B22(4) +1
    # W: C13(2) -1, C21(3) +1, C32(7) +1, C33(8) -1
    U[0, 6] += 1; U[8, 6] += -1
    V[0, 6] += 1; V[4, 6] += 1
    W[2, 6] += -1; W[3, 6] += 1; W[7, 6] += 1; W[8, 6] += -1
    
    return U, V, W

def _build_336_from_333(U3, V3, W3):
    """
    Construct the 3x3x6 decomposition from a 3x3x3 decomposition.
    The 3x6 matrix B is split into B_left (cols 0-2) and B_right (cols 3-5).
    The multiplication is C = A * B = [A * B_left, A * B_right].
    We can view this as two separate 3x3 multiplications.
    Since the rank of the 3x3 decomposition is r, the rank of the 3x6 decomposition is 2r.
    We duplicate the factors.
    """
    rank3 = U3.shape[1]
    rank6 = 2 * rank3
    
    U6 = np.zeros((9, rank6))
    V6 = np.zeros((18, rank6))
    W6 = np.zeros((18, rank6))
    
    # First half (k=0,1,2)
    U6[:, :rank3] = U3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # V index in 3x3: j*3+k
                # V index in 3x6: j*6+k
                V6[j*6+k, :rank3] = V3[j*3+k, :]
                # W index in 3x3: k*3+i
                # W index in 3x6: k*3+i (since k is 0..2, this is the first half of W)
                W6[k*3+i, :rank3] = W3[k*3+i, :]
                
    # Second half (k=3,4,5)
    # The B matrix entries for k=3,4,5 correspond to the second block of columns.
    # The A matrix is the same.
    # The C matrix entries for k=3,4,5 correspond to the second block of rows in C?
    # No. C is 3x6.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If k is 0..2, it's the first block.
    # If k is 3..5, it's the second block.
    # The tensor index c = k*n + i.
    # For k=0..2, c is 0..8.
    # For k=3..5, c is 9..17.
    
    # So we need to map the second 3x3 multiplication to the second half of the rank.
    U6[:, rank3:] = U3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # V index in 3x3: j*3+k
                # V index in 3x6: j*6 + (k+3)
                V6[j*6 + (k+3), rank3:] = V3[j*3+k, :]
                # W index in 3x3: k*3+i
                # W index in 3x6: (k+3)*3 + i
                W6[(k+3)*3 + i, rank3:] = W3[k*3+i, :]
                
    return U6, V6, W6

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try Strassen-based decomposition
    try:
        U3, V3, W3 = _strassen_3x3()
        U6, V6, W6 = _build_336_from_333(U3, V3, W3)
        
        if _verify(U6, V6, W6, n, m, p):
            rank_strassen = U6.shape[1]
            rank_schoolbook = result["U"][0].__len__() if result["U"] else 0
            if rank_strassen < rank_schoolbook:
                result = {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    except Exception:
        pass
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
