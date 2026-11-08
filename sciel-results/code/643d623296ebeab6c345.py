"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k]) * W[k*n+i,c]... 
    Actually correct form: C[i,k] = sum_c W[k*n+i, c] * (U_c . A_flat) * (V_c . B_flat) is WRONG.
    
    Correct: The tensor T_{(i,j),(j',k'),(i'',k'')} = delta(i,i'')*delta(j,j')*delta(k,k'')
    Rank-1: sum_c U_c[i*m+j] * V_c[j'*p+k'] * W_c[k''*n+i'']
    Contract with A[i,j]B[j',k']: output at (i'',k'') = sum_c W_c[k''*n+i''] * (sum_{i,j} U_c[i*m+j]*A[i,j]) * (sum_{j',k'} V_c[j'*p+k']*B[j',k'])
    Wait no - i,j and j',k' are free indices that get summed, but i'' and k'' are the output indices.
    Actually: output[i'',k''] = sum_{i,j,j',k'} T_{(i,j),(j',k'),(i'',k'')} A[i,j] B[j',k']
    = sum_c sum_{i,j,j',k'} U_c[i*m+j] V_c[j'*p+k'] W_c[k''*n+i''] A[i,j] B[j',k']
    = sum_c W_c[k''*n+i''] * (sum_{i,j} U_c[i*m+j] A[i,j]) * (sum_{j',k'} V_c[j'*p+k'] B[j',k'])
    
    Hmm, but this means the output is a sum of rank-1 matrices (outer products of scalars with W_c reshaped).
    Let me verify with schoolbook: U[i*m+j,c]=1 if c corresponds to (i,j,k), V[j*p+k,c]=1, W[k*n+i,c]=1.
    Then for fixed c=(i0,j0,k0): sum_{i,j} U A = A[i0,j0], sum_{j',k'} V B = B[j0,k0], W[k*n+i,c]=1 only when k=k0,i=i0.
    So output[i0,k0] += A[i0,j0]*B[j0,k0]. Summing over all c gives C[i,k] = sum_j A[i,j]B[j,k]. Correct!
    """
    rank = U.shape[1]
    A = np.random.RandomState(42).rand(n, m)
    B = np.random.RandomState(123).rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        s1 = np.dot(Uc, A_flat)
        s2 = np.dot(Vc, B_flat)
        Wc_mat = Wc.reshape(p, n).T  # Wc[k*n+i], reshape to (p,n) then transpose to (n,p)
        C_computed += s1 * s2 * Wc_mat
    
    return np.allclose(C_computed, C_expected, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 54
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Verify schoolbook
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
