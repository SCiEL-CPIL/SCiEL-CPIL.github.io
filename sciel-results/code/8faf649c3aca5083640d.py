import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use NumPy for faster construction and potential future optimization
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill in the schoolbook decomposition
    # Indexing:
    # U row: i*m + k
    # V row: k*p + j
    # W row: j*n + i  (Note: W[k*n+i] in prompt description corresponds to col k (which is j in loop) and row i)
    # Let's re-verify W indexing from seed:
    # Seed: w_idx = j * n + i. 
    # Prompt: W[k*n+i]. In context of C being n x p, C[i,j].
    # The tensor index c corresponds to (j, i) or (i, j)?
    # Seed uses w_idx = j * n + i. This implies the vector W is flattened column-major or similar?
    # If W is (p*n, rank), and index is j*n + i, it maps (j,i) to index.
    # This matches C[j,i] in a column-major layout? Or just a specific flattening order.
    # We must stick to the seed's indexing to be valid.
    
    r_idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                U[u_idx, r_idx] = 1.0
                V[v_idx, r_idx] = 1.0
                W[w_idx, r_idx] = 1.0
                r_idx += 1
                
    # Convert to lists as required by interface? 
    # Interface says: {'U', 'V', 'W'}: factor matrices ... with integer/half-integer entries.
    # The seed returns lists of lists. The evaluator likely accepts numpy arrays or lists.
    # To be safe and match the "list" expectation of the seed's output type strictly if needed, 
    # or to ensure compatibility, we can return lists. 
    # However, the prompt says "factor matrices of shapes...". 
    # Let's return lists to be exactly compatible with the seed's interface type, 
    # but built efficiently.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Optimization Attempt: 
    # Check for trivial rank reductions (e.g. if n, m, p allowed Kronecker products)
    # For 3x4x8, no obvious Kronecker factorization reduces rank below 96 with integer entries easily.
    # We return the schoolbook decomposition which is guaranteed valid.
    
    return {"U": U_list, "V": V_list, "W": W_list}
