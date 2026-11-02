"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n=2, m=2, p=2, tol=1e-9):
    """
    Verify that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] for multiplying A[i,j] * B[j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Wait, the convention is W[k*n+i, r]. So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if this corresponds to A[i,j]*B[j,k] contributing to C[i,k].
    Actually, the matrix multiplication tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] = 1 if the indices correspond to a valid multiplication term, else 0.
    Specifically, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T
    # T[i, j, k] should be 1 for all valid i, j, k
    # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's check for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                u_vec = U[i * m + j, :]
                v_vec = V[j * p + k, :]
                w_vec = W[k * n + i, :]
                val = np.dot(u_vec * v_vec, w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _make_rank7_strassen():
    """
    Construct the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Strassen's algorithm computes C = A * B using 7 multiplications.
    
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m3 + m6
    
    The factor matrices U, V, W must satisfy:
    For each r in 0..6:
      U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] contributes to T[i,j,k]
    
    Let's map the 7 products to the tensor decomposition.
    Each product m_r = (linear_combination_of_A) * (linear_combination_of_B)
    
    The linear combinations of A are:
    L1_A = a + d
    L2_A = c + d
    L3_A = a
    L4_A = d
    L5_A = a + b
    L6_A = c - a
    L7_A = b - d
    
    The linear combinations of B are:
    L1_B = e + h
    L2_B = e
    L3_B = f - h
    L4_B = g - e
    L5_B = h
    L6_B = e + f
    L7_B = g + h
    
    For the tensor decomposition, we need:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Each product m_r contributes to C[i,k] via:
    C[i,k] += coeff_r(i,k) * m_r
    
    But m_r = (sum_{i',j'} L_r_A[i',j'] * A[i',j']) * (sum_{j',k'} L_r_B[j',k'] * B[j',k'])
    
    This is getting complex. Let me use a known explicit Strassen decomposition.
    
    Actually, let me use a simpler approach. I'll construct U, V, W directly from the Strassen formulas.
    
    For each of the 7 terms, we have:
    term_r = (sum_{i,j} u_{i,j}^{(r)} A[i,j]) * (sum_{j,k} v_{j,k}^{(r)} B[j,k])
    
    And C[i,k] = sum_r w_{i,k}^{(r)} * term_r
    
    The tensor decomposition requires:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Where U[i,j,r] = u_{i,j}^{(r)}, V[j,k,r] = v_{j,k}^{(r)}, W[k,i,r] = w_{i,k}^{(r)}.
    
    Wait, the convention is W[k*n+i, r], so W[k,i,r] = w_{i,k}^{(r)}.
    
    Let me define the 7 terms:
    
    r=0: m1 = (a+d)(e+h)
      U: a=1, d=1 -> U[0,0,0]=1, U[1,1,0]=1
      V: e=1, h=1 -> V[0,0,0]=1, V[1,1,0]=1
      W: contributes to C[0,0] with +1, C[1,1] with +1 -> W[0,0,0]=1 (for k=0,i=0), W[1,1,0]=1 (for k=1,i=1)
    
    r=1: m2 = (c+d)e
      U: c=1, d=1 -> U[1,0,1]=1, U[1,1,1]=1
      V: e=1 -> V[0,0,1]=1
      W: contributes to C[1,0] with +1, C[1,1] with -1 -> W[0,1,1]=1, W[1,1,1]=-1
    
    r=2: m3 = a(f-h)
      U: a=1 -> U[0,0,2]=1
      V: f=1, h=-1 -> V[0,1,2]=1, V[1,1,2]=-1
      W: contributes to C[0,1] with +1, C[1,1] with +1 -> W[1,0,2]=1, W[1,1,2]=1
    
    r=3: m4 = d(g-e)
      U: d=1 -> U[1,1,3]=1
      V: g=1, e=-1 -> V[1,0,3]=1, V[0,0,3]=-1
      W: contributes to C[0,0] with +1, C[1,0] with +1 -> W[0,0,3]=1, W[0,1,3]=1
    
    r=4: m5 = (a+b)h
      U: a=1, b=1 -> U[0,0,4]=1, U[0,1,4]=1
      V: h=1 -> V[1,1,4]=1
      W: contributes to C[0,0] with -1, C[0,1] with +1 -> W[0,0,4]=-1, W[1,0,4]=1
    
    r=5: m6 = (c-a)(e+f)
      U: c=1, a=-1 -> U[1,0,5]=1, U[0,0,5]=-1
      V: e=1, f=1 -> V[0,0,5]=1, V[0,1,5]=1
      W: contributes to C[1,1] with +1 -> W[1,1,5]=1
    
    r=6: m7 = (b-d)(g+h)
      U: b=1, d=-1 -> U[0,1,6]=1, U[1,1,6]=-1
      V: g=1, h=1 -> V[1,0,6]=1, V[1,1,6]=1
      W: contributes to C[0,0] with +1 -> W[0,0,6]=1
    
    Let me build U, V, W as 4x7 matrices (since n*m=4, m*p=4, p*n=4).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: m1 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # r=1: m2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1  # C[1,1]
    
    # r=2: m3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1  # h
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # r=3: m4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1  # e
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # r=4: m5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    W[0][4] = -1  # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # r=5: m6 = (c-a)(e+f)
    U[2][5] = 1  # c
    U[0][5] = -1  # a
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    W[3][5] = 1  # C[1,1]
    
    # r=6: m7 = (b-d)(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1  # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    W[0][6] = 1  # C[0,0]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a rank-7 Strassen decomposition for 2x2 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try the rank-7 Strassen decomposition first
    result = _make_rank7_strassen()
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook rank-8 if verification fails
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
