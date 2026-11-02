import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def verify_decomposition(U, V, W, n=2, m=2, p=8):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Test with random matrices
    for _ in range(5):
        A = np.random.randint(-5, 6, (n, m)).astype(float)
        B = np.random.randint(-5, 6, (m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for t in range(rank):
            a_vec = A.flatten()  # n*m elements
            v_vec = V[:, t]  # m*p elements
            w_vec = W[:, t]  # p*n elements
            
            # u_t^T * A_flat
            a_val = np.dot(U[:, t], a_vec)
            
            # v_t^T * B_flat
            b_val = np.dot(v_vec, B.flatten())
            
            # w_t^T * C_flat (this is the coefficient for each C[i,k])
            c_contrib = a_val * b_val * w_vec
            C_computed += c_vec_to_matrix(c_contrib, n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def c_vec_to_matrix(vec, n, p):
    """Convert flattened C vector (index k*n+i) to 2D matrix."""
    mat = np.zeros((n, p))
    for k in range(p):
        for i in range(n):
            mat[i, k] = vec[k * n + i]
    return mat

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with Strassen-based 28-rank decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen products for 2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        for r, prod in enumerate(products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff

    # Verify the decomposition
    if not verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if Strassen fails
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j, t] = 1
                    V[j * p + k, t] = 1
                    W[k * n + i, t] = 1
    
    # Try to reduce rank by looking for linear dependencies
    # Check if any columns are linearly dependent
    if time.time() < deadline - 10:
        U_float = U.astype(float)
        V_float = V.astype(float)
        W_float = W.astype(float)
        
        # Check rank of U, V, W matrices
        U_rank = np.linalg.matrix_rank(U_float)
        V_rank = np.linalg.matrix_rank(V_float)
        W_rank = np.linalg.matrix_rank(W_float)
        
        # If any factor has lower rank than the nominal rank, we might be able to reduce
        # But we need all three to have full column rank for the decomposition to be minimal
        
        # Try to find if we can remove terms by checking if removing a term still works
        for t_remove in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            U_test = np.delete(U, t_remove, axis=1)
            V_test = np.delete(V, t_remove, axis=1)
            W_test = np.delete(W, t_remove, axis=1)
            
            if verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
