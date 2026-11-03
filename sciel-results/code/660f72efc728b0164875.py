import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    The approach uses 8 independent 2x2 Strassen blocks (rank 7 each) for a total rank of 56.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56
    
    # Allocate factor matrices
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    # Strassen coefficients for 2x2 multiplication
    # Each term: (u_indices, v_indices, w_indices)
    # where each is a list of (local_index, coefficient)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # M0 = (A00 + A11) * (B00 + B11) -> C00 += M0, C11 += M0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M1 = (A10 + A11) * B00 -> C10 += M1, C11 -= M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M2 = A00 * (B10 - B11) -> C01 += M2, C11 += M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M3 = A11 * (B01 - B00) -> C00 += M3, C10 += M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M4 = (A00 + A01) * B11 -> C00 -= M4, C01 += M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M5 = (A10 - A00) * (B00 + B01) -> C11 += M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M6 = (A01 - A11) * (B10 + B11) -> C00 += M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Process 8 blocks: 2 in j-dimension (h=0,1) x 4 in k-dimension (s=0,1,2,3)
    for h in range(2):
        for s in range(4):
            block_idx = h * 4 + s
            base_rank = block_idx * 7
            
            # Compute global indices for this 2x2 block
            # A[i,j] flattened as i*4+j
            # B[j,k] flattened as j*8+k  
            # C[k,i] flattened as k*2+i (trace convention)
            
            a_idx = [i * 4 + (2 * h + j) for i in range(2) for j in range(2)]
            b_idx = [(2 * h + j) * 8 + (2 * s + k) for j in range(2) for k in range(2)]
            c_idx = [(2 * s + k) * 2 + i for k in range(2) for i in range(2)]
            
            # Fill in the Strassen decomposition for this block
            for t in range(7):
                r = base_rank + t
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                
                for local_i, coeff in u_coeffs:
                    U[a_idx[local_i], r] = coeff
                for local_i, coeff in v_coeffs:
                    V[b_idx[local_i], r] = coeff
                for local_i, coeff in w_coeffs:
                    W[c_idx[local_i], r] = coeff
            
            # Time check
            if time.time() - start_time > 130:
                break
        
        if time.time() - start_time > 130:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
