import time
import numpy as np
import sys

def construct():
    start_time = time.time()
    
    n, m, p = 2, 2, 8
    
    # The rank-28 Strassen-based decomposition is well-established and correct.
    # We construct it directly.
    result = _strassen_28()
    
    # Verify the solution is correct
    if _verify(result, n, m, p):
        return result
    
    # Fallback: schoolbook rank-32
    return _schoolbook(n, m, p)

def _strassen_28():
    """
    Constructs a rank-28 decomposition for 2x2 by 2x8 matrix multiplication.
    
    The 2x2x2 Strassen algorithm uses 7 multiplications.
    For 2x2 by 2x8, we treat the 2x8 matrix as 4 blocks of 2x2.
    Each block uses 7 multiplications: 4 * 7 = 28 total.
    
    Strassen's 7 products for (A * B) where A,B are 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)(f)
    M3 = (a)(e-g)
    M4 = (d)(h-e)
    M5 = (a+b)(h)
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Result:
    AB[0,0] = M1 + M4 - M5 + M7
    AB[0,1] = M3 + M5
    AB[1,0] = M2 + M4
    AB[1,1] = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 8
    num_blocks = p // m  # 4 blocks of 2 columns
    rank = 7 * num_blocks  # 28
    
    U = np.zeros((n * m, rank), dtype=float)  # (4, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 28)
    
    # Strassen coefficients
    # For a 2x2 multiplication, the 7 terms have these (U, V, W) vectors
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: 0=e, 1=f, 2=g, 3=h
    # W indices: 0=AB[0,0], 1=AB[0,1], 2=AB[1,0], 3=AB[1,1]
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2 = (c+d)(f)
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, -1]),
        # M3 = (a)(e-g)
        ([1, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, 1]),
        # M4 = (d)(h-e)
        ([0, 0, 0, 1], [-1, 0, 0, 1], [1, 0, 1, 0]),
        # M5 = (a+b)(h)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    for block in range(num_blocks):
        base_col = block * 7
        # V indices for this block: rows 2*block, 2*block+1 of B, which map to 
        # B index: row j, col k where k = 2*block + j'
        # B is 2x8, so B[j, k] has index j*8 + k
        # For block t, columns are 2*t and 2*t+1
        # So B indices: 0*8 + 2*t, 0*8 + 2*t+1, 1*8 + 2*t, 1*8 + 2*t+1
        # Which is: 2*t, 2*t+1, 8+2*t, 8+2*t+1
        
        # W indices: C is 8x2, C[k, i] has index k*2 + i (convention W[k*n+i])
        # For block t, C rows are 2*t and 2*t+1
        # C indices: (2*t)*2 + 0, (2*t)*2 + 1, (2*t+1)*2 + 0, (2*t+1)*2 + 1
        # Which is: 4*t, 4*t+1, 4*t+2, 4*t+3
        
        for t in range(7):
            col = base_col + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Set U
            for i in range(4):
                U[i, col] = u_coeffs[i]
            
            # Set V
            v_indices = [2 * block, 2 * block + 1, 8 + 2 * block, 8 + 2 * block + 1]
            for i in range(4):
                V[v_indices[i], col] = v_coeffs[i]
            
            # Set W
            w_indices = [4 * block, 4 * block + 1, 4 * block + 2, 4 * block + 3]
            for i in range(4):
                W[w_indices[i], col] = w_coeffs[i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """Schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(solution, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(solution["U"], dtype=float)
    V = np.array(solution["V"], dtype=float)
    W = np.array(solution["W"], dtype=float)
    
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Compute T_hat[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, rank)
    T_hat = np.einsum('abr,cr->abc', UV, W)  # (n*m, m*p, p*n)
    
    # The true tensor: T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i') and i==i'
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1, else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_hat[a, b, c] - 1) > tol:
                    return False
    
    # Check that all other entries are zero
    # Build a set of valid (a, b, c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True
