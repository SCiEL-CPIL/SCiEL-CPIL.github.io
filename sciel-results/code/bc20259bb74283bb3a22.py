"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # We'll use a known low-rank decomposition for 2x3 * 3x8
    # First, let's construct a decomposition using the bilinear structure.
    # The matrix multiplication tensor T has components T_{i,j,k,l} = 1 if we're multiplying A[i,j]*B[j,k] -> C[i,k]
    # For 2x3 * 3x8, we have i in {0,1}, j in {0,1,2}, k in {0..7}
    
    # One approach: use the fact that we can decompose by splitting the middle dimension.
    # But a better approach is to find an explicit low-rank decomposition.
    
    # Let me try to construct one manually or use a known result.
    # For m=3, we can use the fact that the rank of matrix multiplication 2x3 * 3x8
    # is at most 2*3*8 = 48 (schoolbook), but can be lower.
    
    # A known technique: if we can write the multiplication as a sum of rank-1 tensors
    # with fewer terms than n*m*p, we get a lower rank.
    
    # Let's try to construct a decomposition. The tensor is:
    # T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*
    # where A_{i,j} is the basis for input A, B_{j,k} for input B, C_{i,k}^* for output C.
    
    # A rank-r decomposition would be:
    # T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # For each t, u_t[i*m+j], v_t[j*p+k], w_t[k*n+i]
    # The condition is: sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1 if this is the (i,j,k) term
    # and 0 otherwise.
    
    # Let me try to find such a decomposition with small r.
    
    # Start with r=1 and increase. But this is a hard search. Let me try a structured approach.
    
    # Actually, let me try a different approach: use the fact that we can combine terms.
    # For example, if we fix j, we have n*p = 2*8 = 16 terms for each j.
    # For j=0,1,2, we have 48 terms total.
    
    # Can we share structure between different j values?
    
    # Let me try to use the following idea: 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a linear combination for each (i,k).
    
    # One way to get a lower rank: if some of the "gates" (i,j,k) can be combined
    # using the same rank-1 term.
    
    # Let me try to construct a decomposition with r = 24 or so.
    
    # Actually, let me just use the schoolbook decomposition as a fallback,
    # but first try to see if there's a known better bound.
    
    # For now, let me construct the schoolbook decomposition:
    rank = n * m * p  # 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists of lists for the expected format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
