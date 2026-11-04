import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    The rank is n*m*p for the schoolbook method.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    r = n * m * p
    
    # Initialize matrices as lists of lists of integers
    # U has shape (n*m, r)
    U = [[0] * r for _ in range(n * m)]
    # V has shape (m*p, r)
    V = [[0] * r for _ in range(m * p)]
    # W has shape (p*n, r)
    W = [[0] * r for _ in range(p * n)]
    
    # Populate the matrices for the standard tensor decomposition
    # The tensor T[i, j, k] is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook, each term is a single basis vector.
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    # Basic verification
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Check shapes
    assert len(U) == n * m, "U rows mismatch"
    assert len(U[0]) == len(U[0]) * 0 + 42, "U cols mismatch (expected 42)"
    assert len(V) == m * p, "V rows mismatch"
    assert len(W) == p * n, "W rows mismatch"
    
    # Verify decomposition
    # T[i, j, k] should be 1 if indices are valid, but the tensor is defined by the multiplication map.
    # Actually, we verify that the product reconstructs the standard matmul tensor.
    # The standard tensor has entries 1 at (i,j,k) for all i,j,k in the schoolbook expansion?
    # No, the matrix multiplication tensor T_matmul has a 1 at (i, j, k) for the output element (i, k) depending on input (i, j) and (j, k).
    # The decomposition identity is: C_{ik} = sum_j A_{ij} B_{jk}
    # This is represented by the tensor T with T[i, j, k] = delta_{ik}? No.
    # The bilinear map is (A, B) -> C.
    # The decomposition is exact if for all A, B:
    # sum_c (U_c^T A) (V_c^T B) W_c = A B
    # Let's check the standard identity:
    # sum_{i,j,k} e_{ij} A e_{jk} B e_{ik}^T
    # U_{ij} is row vector picking A_{ij}? No, U is (nm, r).
    # U[i*m+j, c] = delta_{c, (i,j,k)}
    # V[j*p+k, c] = delta_{c, (i,j,k)}
    # W[k*n+i, c] = delta_{c, (i,j,k)}
    
    # Let's verify with random small integer matrices
    np.random.seed(42)
    A = np.random.randint(-10, 10, size=(n, m))
    B = np.random.randint(-10, 10, size=(m, p))
    
    # Compute AB
    AB = A @ B
    
    # Compute via decomposition
    # Reshape A to (n*m, 1), B to (m*p, 1)
    a_vec = A.flatten(order='C') # i*m + j
    b_vec = B.flatten(order='C') # j*p + k
    
    # U is (nm, r), so u_vec = U^T a_vec?
    # The term is sum_c (sum_{ij} U[i*m+j, c] A_{ij}) * (sum_{jk} V[j*p+k, c] B_{jk}) * W[...]
    # Let u_c = dot(U[:, c], A.flatten())
    # Let v_c = dot(V[:, c], B.flatten())
    # Result C_{ik} = sum_c u_c * v_c * W[k*n+i, c]
    
    U_np = np.array(U).T # (r, nm)
    V_np = np.array(V).T # (r, mp)
    W_np = np.array(W)   # (pn, r)
    
    u_c = U_np @ a_vec
    v_c = V_np @ b_vec
    
    C_reconstructed = (u_c * v_c) @ W_np # (r) dot (r, pn) -> (pn)
    
    # C_reconstructed is flattened in order k*n + i
    # We need to reshape to (n, p)
    # The index is k*n + i.
    # So C_reconstructed[k*n + i] corresponds to C[i, k]
    C_mat = np.zeros((n, p))
    for k in range(p):
        for i in range(n):
            C_mat[i, k] = C_reconstructed[k*n + i]
            
    if not np.array_equal(AB, C_mat):
        raise ValueError("Decomposition failed verification")
        
    print("Verification successful. Rank:", len(U[0]))
