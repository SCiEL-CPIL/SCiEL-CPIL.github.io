"""Matrix multiplication 2x4x7 - rank-48 decomposition using Strassen blocks."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We need to decompose the 2x4x7 matrix multiplication tensor.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r]
    
    # Strategy:
    # Split into blocks:
    # 1. 2x4x4 part (j=0..3, k=0..3): 4 Strassen blocks (2x2x2) -> 4 * 7 = 28
    # 2. 2x4x2 part (j=0..3, k=4..5): 2 Strassen blocks (2x2x2) -> 2 * 7 = 14
    # 3. 2x4x1 part (j=0..3, k=6): 8 rank-1 terms -> 8
    # Total: 28 + 14 + 8 = 50
    
    # Can we do better?
    # For the 2x4x1 part, C[i,6] = sum_{j=0}^3 A[i,j] * B[j,6]
    # This is a 2x4x1 tensor. The rank of this tensor is 4 (since it's essentially
    # two rows of a 4-vector). Wait, no.
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6] + A[0,2]B[2,6] + A[0,3]B[3,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6] + A[1,2]B[2,6] + A[1,3]B[3,6]
    # This is a sum of 4 rank-1 tensors:
    # For each j, we have A[:,j] * B[j,6] * e_j^T (in the A dimension)
    # Actually, the tensor T_{i,j,k} = delta_{k,6} * A_{i,j} B_{j,k}
    # This tensor has rank 4. Each term is A[i,j] * B[j,6] for fixed j.
    # So we need 4 terms for i=0,1 and j=0..3.
    # Wait, the rank-1 term is u_i * v_j * w_k.
    # For fixed j, the term is A[i,j] * B[j,6].
    # This can be written as sum_j (A[:,j]) (B[j,6]) (e_k where k=6)
    # So for each j, we have one rank-1 term that contributes to both i=0 and i=1.
    # Let's check:
    # Term j: U[i, j] = A[i,j], V[j, 6] = B[j,6], W[6, i] = 1
    # Then sum_j U[i,j] V[j,6] W[6,i] = sum_j A[i,j] B[j,6] * 1 = C[i,6]
    # Yes! So we only need 4 terms for the 2x4x1 part, not 8.
    # 
    # Let's re-verify the index conventions.
    # U has shape (n*m, rank). Index is i*m + j.
    # V has shape (m*p, rank). Index is j*p + k.
    # W has shape (p*n, rank). Index is k*n + i.
    #
    # Term j:
    # U[i*m + j] += 1  (for i=0,1) -> Wait, U is a matrix.
    # The decomposition is C = sum_r U[:,r] V[:,r]^T ... no, it's an outer product of vectors.
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    #
    # For the 2x4x1 part (k=6):
    # We want T_{i,j,6} = A_{i,j} B_{j,6}.
    # Let's use 4 terms, one for each j.
    # Term r_j (r = 28 + 14 + j = 42 + j):
    # U[i*m + j, r_j] = 1 for i=0,1? No, that would make U[i*m+j, r_j] = 1 for all i.
    # Then sum_i U[i*m+j, r_j] ... wait, the sum is over r.
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    #
    # If we set for term r_j:
    # U[i*m + j, r_j] = 1 for i=0,1?
    # V[j*p + 6, r_j] = 1
    # W[6*n + i, r_j] = 1 for i=0,1?
    #
    # Then T_{i,j,6} = sum_r U_{i*m+j, r} V_{j*p+6, r} W_{6*n+i, r}
    # = U_{i*m+j, r_j} * V_{j*p+6, r_j} * W_{6*n+i, r_j}
    # = 1 * 1 * 1 = 1.
    # But we want A_{i,j} B_{j,6}.
    # The decomposition is linear in A and B.
    # The coefficients U, V, W are constants.
    # So T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # This sum must equal A_{i,j} B_{j,k} for all A, B.
    # This means the tensor defined by U, V, W must be exactly the multiplication tensor.
    # The multiplication tensor has entries T_{i,j,k} = delta_{j, ?} ... no.
    # The multiplication tensor M_{i,j,k} is 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # It is 1 for all i,j,k such that the index is valid.
    # Specifically, M_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, no. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the term A_{i,j} B_{j,k} has coefficient 1 in C_{i,k}.
    # So the tensor is T_{i,j,k} = 1 for all valid i,j,k.
    #
    # So we need sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k.
    #
    # Let's re-evaluate the 2x4x1 part.
    # We need sum_r U_{i*m+j, r} V_{j*p+6, r} W_{6*n+i, r} = 1 for all i in {0,1}, j in {0,1,2,3}.
    #
    # If we use 4 terms, one for each j:
    # For term r_j:
    # U[i*m + j, r_j] = 1 for i=0,1?
    # V[j*p + 6, r_j] = 1
    # W[6*n + i, r_j] = 1 for i=0,1?
    #
    # Then for fixed i,j:
    # sum_r ... = sum_{j'} delta_{j,j'} * 1 * 1 = 1.
    # Yes, this works!
    #
    # So the rank for the 2x4x1 part is 4, not 8.
    #
    # Total rank: 28 + 14 + 4 = 46.
    #
    # Let's double check if we can save more.
    # The 2x4x4 part uses 4 Strassen blocks. Can we do better?
    # 2x4x4 is equivalent to 2x2x4 + 2x2x4.
    # Each 2x2x4 is 2x2x2 + 2x2x2.
    # So 4 Strassen blocks seems optimal for this block structure.
    #
    # What about the 2x4x2 part?
    # 2x4x2 = 2x2x2 + 2x2x2.
    # 2 Strassen blocks.
    #
    # Can we combine the last part (k=6) with the previous parts?
    # The k=6 part is C[i,6] = sum_j A[i,j] B[j,6].
    # The k=4,5 parts are C[i,4] and C[i,5].
    # They are independent in terms of output indices.
    #
    # So rank 46 seems achievable.
    #
    # Let's implement rank 46.
    
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks
    # Block (i=0, j=0..1, k=0..1)
    add_strassen(0, 0, 0)
    # Block (i=0, j=0..1, k=2..3)
    add_strassen(0, 0, 2)
    # Block (i=0, j=2..3, k=0..1)
    add_strassen(0, 2, 0)
    # Block (i=0, j=2..3, k=2..3)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks
    # Block (i=0, j=0..1, k=4..5)
    add_strassen(0, 0, 4)
    # Block (i=0, j=2..3, k=4..5)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1 (k=6) - 4 rank-1 terms
    # For each j in 0..3, we have one term
    for j in range(4):
        # U[i*m + j, col] = 1 for i=0,1
        U[0*m + j, col] = 1
        U[1*m + j, col] = 1
        # V[j*p + 6, col] = 1
        V[j*p + 6, col] = 1
        # W[6*n + i, col] = 1 for i=0,1
        W[6*n + 0, col] = 1
        W[6*n + 1, col] = 1
        col += 1
        
    # Time guard
    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
