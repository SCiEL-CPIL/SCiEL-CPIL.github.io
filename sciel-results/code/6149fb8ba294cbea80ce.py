"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_decomposition(n, m, p):
    """
    Recursively build a tensor decomposition for matrix multiplication of size n x m x p
    using Strassen's algorithm when dimensions are powers of 2, falling back to schoolbook
    for base cases.
    
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) as lists of lists.
    """
    n, m, p = int(n), int(m), int(p)
    
    # For the task we need n=m=p=6. We'll pad to 8x8x8 and then project.
    # Actually, let's think differently. The tensor for 6x6x6 matrix multiplication
    # is a sub-tensor of the 8x8x8 tensor if we embed it properly? No, that's not right.
    # 
    # A better approach: The matrix multiplication tensor T_{n,m,p} can be decomposed
    # using the recursive structure. For n=m=p=6, we can use the fact that 6 < 8 = 2^3.
    # We can use a rank-7 decomposition for 2x2x2 (Strassen) and then build up.
    #
    # Actually, the cleanest way: use the known rank-7 decomposition for 2x2x2.
    # Then for 4x4x4, the rank is at most 7^2 = 49 (but actually it's known to be 49 for 4x4x4? 
    # No, the rank of 4x4x4 matmul is at most 49 via Strassen recursion, but the exact rank is unknown, 
    # believed to be 49 or less).
    # For 8x8x8, rank is at most 7^3 = 343.
    #
    # But we need 6x6x6. One approach: embed 6x6x6 into 8x8x8, get a rank-343 decomposition,
    # then restrict to the 6x6x6 sub-tensor. This gives a valid decomposition of rank at most 343.
    #
    # Even better: we can directly construct a Strassen-based decomposition for 6x6x6 by
    # splitting 6 = 4 + 2 or similar, but that gets complicated.
    #
    # Let's use the 8x8x8 approach: 
    # 1. Build the 8x8x8 Strassen decomposition (rank 343).
    # 2. The 6x6x6 tensor is obtained by zeroing out indices >= 6 in each mode.
    # 3. The same rank-343 factors, when restricted, give a valid decomposition.
    #
    # Actually, we can do even better. Let's directly build a recursive Strassen decomposition
    # that handles arbitrary sizes by padding to the next power of 2.
    
    # Pad to next power of 2
    def next_pow2(x):
        p2 = 1
        while p2 < x:
            p2 *= 2
        return p2
    
    N = next_pow2(max(n, m, p))
    
    # Build Strassen decomposition for N x N x N
    # Base case: 1x1x1 -> rank 1
    if N == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    # For N=2, use Strassen's 7-term decomposition
    # For N>2, recursively build for N/2 and combine
    
    # We'll build it recursively
    def build_strassen(size):
        """Build Strassen decomposition for size x size x size."""
        if size == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        half = size // 2
        U_half, V_half, W_half = build_strassen(half)
        rank_half = U_half.shape[1]
        
        # Strassen's 7 products for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each Mi is a matrix product of size half x half, so we can use the recursive decomposition.
        # Total rank = 7 * rank_half
        
        rank = 7 * rank_half
        U = np.zeros((size * size, rank))
        V = np.zeros((size * size, rank))
        W = np.zeros((size * size, rank))
        
        # Define the 7 Strassen combinations
        # For each of the 7 products, we need to define how A and B are preprocessed
        # and how the result is combined into C.
        
        # Indices for quadrants:
        # A11: [0:half, 0:half], A12: [0:half, half:size]
        # A21: [half:size, 0:half], A22: [half:size, half:size]
        # Similarly for B and C.
        
        def idx(r, c, size, half):
            """Map (r, c) in size x size to linear index."""
            return r * size + c
        
        def quad_indices(size, half):
            """Return row and col index ranges for the 4 quadrants."""
            return [
                (range(0, half), range(0, half)),       # 11
                (range(0, half), range(half, size)),    # 12
                (range(half, size), range(0, half)),    # 21
                (range(half, size), range(half, size)), # 22
            ]
        
        quads = quad_indices(size, half)
        
        # For each of the 7 Strassen products, define the A-combination and B-combination
        # as sparse matrices, and the C-combination as a sparse matrix.
        
        # We'll build U, V, W by iterating over the 7 products and the rank_half sub-decomposition.
        
        for prod in range(7):
            # Define the A-preprocessing matrix (size*size x size*size) 
            # and B-preprocessing matrix (size*size x size*size)
            # and C-postprocessing matrix (size*size x size*size)
            
            A_comb = np.zeros((size * size, size * size))
            B_comb = np.zeros((size * size, size * size))
            C_comb = np.zeros((size * size, size * size))
            
            if prod == 0:  # M1 = (A11+A22)(B11+B22)
                for r, c in zip(quads[0][0], quads[0][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    C_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 1:  # M2 = (A21+A22)B11
                for r, c in zip(quads[2][0], quads[2][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    C_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[2][0], quads[2][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 2:  # M3 = A11(B21-B22)
                for r, c in zip(quads[0][0], quads[0][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[2][0], quads[2][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    B_comb[r * size + c, r * size + c] = -1
                for r, c in zip(quads[1][0], quads[1][1]):
                    C_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 3:  # M4 = A22(B11-B21)
                for r, c in zip(quads[3][0], quads[3][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[2][0], quads[2][1]):
                    B_comb[r * size + c, r * size + c] = -1
                for r, c in zip(quads[0][0], quads[0][1]):
                    C_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[2][0], quads[2][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 4:  # M5 = (A11+A12)B22
                for r, c in zip(quads[0][0], quads[0][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[1][0], quads[1][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    C_comb[r * size + c, r * size + c] = -1
                for r, c in zip(quads[1][0], quads[1][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 5:  # M6 = (A21-A11)(B11+B12)
                for r, c in zip(quads[2][0], quads[2][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    A_comb[r * size + c, r * size + c] = -1
                for r, c in zip(quads[0][0], quads[0][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[1][0], quads[1][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    C_comb[r * size + c, r * size + c] = 1
            elif prod == 6:  # M7 = (A12-A22)(B21+B22)
                for r, c in zip(quads[1][0], quads[1][1]):
                    A_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    A_comb[r * size + c, r * size + c] = -1
                for r, c in zip(quads[2][0], quads[2][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[3][0], quads[3][1]):
                    B_comb[r * size + c, r * size + c] = 1
                for r, c in zip(quads[0][0], quads[0][1]):
                    C_comb[r * size + c, r * size + c] = 1
            
            # Now, for this product, the effective A matrix is A_comb @ A, and B is B_comb @ B.
            # The product is (A_comb @ A) @ (B_comb @ B) = A_comb @ (A @ B) @ B_comb... no.
            # Actually, the product is (A_comb @ A) @ (B_comb @ B) where A and B are size x size.
            # This equals A_comb @ A @ B_comb @ B, which is not a simple matmul.
            # 
            # Wait, I'm overcomplicating. Let me think again.
            # 
            # In Strassen, for the 2x2 case, we compute 7 scalar products of 2x2 submatrices.
            # Each "product" is a matrix multiplication of two half-size matrices.
            # So M1 = A11_plus_A22 @ B11_plus_B22, where A11_plus_A22 is a half x half matrix.
            # 
            # So the rank of the full decomposition is 7 * (rank of half x half matmul).
            # 
            # To build the factors:
            # For each of the 7 products, and for each of the rank_half factors in the sub-decomposition,
            # we create a column in U, V, W.
            # 
            # The U factor for product t and sub-factor s is:
            # U[i*m+j, t*rank_half+s] = (A_comb_t @ U_half)[i*m+j, s]
            # But A_comb_t is a permutation/sign matrix, so it just picks and signs the relevant entries.
            # 
            # Let me just build it directly.
            
            # For product t, the A-side combination picks certain quadrants with signs.
            # We can represent this as a sparse operation.
            
            # Actually, let's just build the full U, V, W matrices directly.
            
            # For each of the 7 products, we have a specific combination of A and B submatrices.
            # The resulting C is a specific combination of C submatrices.
            
            # Let me define the combinations more carefully.
            
            pass
        
        # This is getting complex. Let me use a cleaner recursive approach.
        
        # Build the decomposition for size x size x size using 7 recursive calls.
        # Each recursive call handles a half x half x half matmul.
        
        U = np.zeros((size * size, rank))
        V = np.zeros((size * size, rank))
        W = np.zeros((size * size, rank))
        
        # The 7 Strassen products, defined by which quadrants of A, B, C are involved and with what signs.
        # Each product is: (sum of signed A-quadrants) @ (sum of signed B-quadrants) -> sum of signed C-quadrants
        
        strassen_products = [
            # (A_quadrants_with_signs, B_quadrants_with_signs, C_quadrants_with_signs)
            # Quadrant 0=11, 1=12, 2=21, 3=22
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
            ([(2, 1), (3, 1)], [(0, 1)], [(3, 1), (2, 1)]),  # M2
            ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),  # M3
            ([(3, 1)], [(0, 1), (2, -1)], [(0, 1), (2, 1)]),  # M4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),  # M5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),  # M6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),  # M7
        ]
        
        for prod_idx, (a_comb, b_comb, c_comb) in enumerate(strassen_products):
            base_col = prod_idx * rank_half
            
            # Build the A-preprocessing: for each row of U_half, determine which A-entries it corresponds to
            # and with what sign.
            # U_half has shape (half*half, rank_half). Each column s of U_half corresponds to a factor.
            # For the full size, U[i*size+j, base_col+s] = sum over (q, sign) in a_comb of sign * U_half[i_q*size+j_q, s]
            # where i_q, j_q are the row/col indices in the half-size matrix corresponding to quadrant q.
            
            # Map quadrant to (row_offset, col_offset) in half-size
            quad_map = {
                0: (0, 0),   # 11
                1: (0, half), # 12
                2: (half, 0), # 21
                3: (half, half), # 22
            }
            
            for s in range(rank_half):
                # Accumulate U
                for (q, sign) in a_comb:
                    ro, co = quad_map[q]
                    for i in range(half):
                        for j in range(half):
                            full_i = i + ro
                            full_j = j + co
                            U[full_i * size + full_j, base_col + s] += sign * U_half[i * half + j, s]
                
                # Accumulate V
                for (q, sign) in b_comb:
                    ro, co = quad_map[q]
                    for i in range(half):
                        for j in range(half):
                            full_i = i + ro
                            full_j = j + co
                            V[full_i * size + full_j, base_col + s] += sign * V_half[i * half + j, s]
                
                # Accumulate W
                for (q, sign) in c_comb:
                    ro, co = quad_map[q]
                    for i in range(half):
                        for j in range(half):
                            full_i = i + ro
                            full_j = j + co
                            W[full_i * size + full_j, base_col + s] += sign * W_half[i * half + j, s]
        
        return U, V, W
    
    U_full, V_full, W_full = build_strassen(N)
    
    # Now restrict to n x m x p by zeroing out entries with indices >= n, m, p respectively.
    # U has shape (N*N, rank). We keep rows where row_idx < n*N and the row corresponds to i < n, j < m.
    # Actually, U is indexed by (i, j) where i in [0,N), j in [0,N). We keep (i,j) where i < n and j < m.
    
    U = np.zeros((n * m, U_full.shape[1]))
    V = np.zeros((m * p, V_full.shape[1]))
    W = np.zeros((p * n, W_full.shape[1]))
    
    for i in range(n):
        for j in range(m):
            U[i * m + j] = U_full[i * N + j]
    
    for j in range(m):
        for k in range(p):
            V[j * p + k] = V_full[j * N + k]
    
    for k in range(p):
        for i in range(n):
            W[k * n + i] = W_full[k * N + i]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    try:
        U, V, W = _strassen_decomposition(n, m, p)
        rank = len(U[0]) if U else 0
        
        # Verify correctness
        if len(U) > 0 and len(U[0]) > 0:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if i==k and j==j... 
            # Actually, the matmul tensor T[i,j,k] = 1 if we're computing (A@B)[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] = 1 for all i,j,k in the "diagonal" sense... no.
            # The matrix multiplication tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
            # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the appropriate sense).
            # 
            # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p).
            
            # Compute the reconstructed tensor
            # This is a 3D tensor of size n x m x p
            # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # For efficiency, compute using einsum or matrix operations
            # Let's do it naively for small sizes
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                return {"U": U, "V": V, "W": W}
            else:
                # Fall back to schoolbook
                pass
        else:
            pass
    except Exception:
        pass
    
    # Schoolbook fallback
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
