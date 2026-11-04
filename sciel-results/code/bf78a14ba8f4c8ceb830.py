import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Target: Multiply A (n x m) by B (m x p) -> C (n x p)
    # Tensor indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We split the p dimension into blocks.
    # Since n=4, m=4, we can use Strassen for 4x4x4 blocks.
    # p=9 can be split into 4 + 4 + 1.
    
    block_sizes = [4, 4, 1]
    
    # Generate Strassen factors for 4x4x4
    # Strassen recursion:
    # S1 = (A11 + A22) * (B11 + B22)
    # S2 = (A21 + A22) * B11
    # S3 = A11 * (B12 - B22)
    # S4 = A22 * (B21 - B11)
    # S5 = (A11 + A12) * B22
    # S6 = (A21 - A11) * (B11 + B12)
    # S7 = (A12 - A22) * (B21 + B22)
    
    # We need the tensor decomposition coefficients for Strassen.
    # For N=2, the rank is 7.
    # For N=4, we apply Strassen to each quadrant, so rank is 7 * 7 = 49.
    
    def get_strassen_decomposition(N):
        """
        Returns U, V, W matrices for N x N x N matrix multiplication.
        U shape: (N*N, rank)
        V shape: (N*N, rank)
        W shape: (N*N, rank)
        
        Indices:
        U[i*N+j, c]
        V[j*N+k, c]
        W[k*N+i, c]
        """
        if N == 1:
            return [[1]], [[1]], [[1]]
        
        # Recursively get decomposition for N/2
        half = N // 2
        U_half, V_half, W_half = get_strassen_decomposition(half)
        rank_half = len(U_half[0])
        
        # Strassen 7 products for 2x2 block of NxN matrices
        # Each "product" is a multiplication of two (N/2)x(N/2) matrices.
        # We combine the half-decompositions.
        
        # The 7 linear combinations for A and B:
        # S1: A11+A22, B11+B22
        # S2: A21+A22, B11
        # S3: A11, B12-B22
        # S4: A22, B21-B11
        # S5: A11+A12, B22
        # S6: A21-A11, B11+B12
        # S7: A12-A22, B21+B22
        
        # For each product t in 0..6:
        # We use the rank_half decomposition for that sub-multiplication.
        # Total rank = 7 * rank_half
        
        rank = 7 * rank_half
        U = [[0] * rank for _ in range(N*N)]
        V = [[0] * rank for _ in range(N*N)]
        W = [[0] * rank for _ in range(N*N)]
        
        # Map local (i, j) in N/2 to global (I, J) in N
        # I = i if i < half else i? No, standard indexing:
        # A11 is [0:half, 0:half]
        # A12 is [0:half, half:N]
        # A21 is [half:N, 0:half]
        # A22 is [half:N, half:N]
        
        # Helper to map local index (li, lj) in half-sized block 
        # to global row/col indices based on block position (br, bc)
        # where br, bc in {0,1}
        def map_idx(li, lj, br, bc):
            gi = br * half + li
            gj = bc * half + lj
            return gi, gj
        
        # For each of the 7 Strassen terms
        # We need to define which quadrants of A and B are involved,
        # and with what signs.
        
        # Term 0: S1 = (A11+A22)*(B11+B22)
        # Term 1: S2 = (A21+A22)*B11
        # Term 2: S3 = A11*(B12-B22)
        # Term 3: S4 = A22*(B21-B11)
        # Term 4: S5 = (A11+A12)*B22
        # Term 5: S6 = (A21-A11)*(B11+B12)
        # Term 6: S7 = (A12-A22)*(B21+B22)
        
        # Structure:
        # A_quadrants: list of (row_block, col_block, sign)
        # B_quadrants: list of (row_block, col_block, sign)
        # C_combination: list of (row_block, col_block, sign) for the output
        
        # Actually, it's easier to think in terms of the tensor factors.
        # For each Strassen term t:
        # U contribution comes from the A side combination.
        # V contribution comes from the B side combination.
        # W contribution comes from the C side combination (sum of quadrants).
        
        # Let's define the combinations explicitly for N=2 case (base of recursion)
        # For the recursive step, we treat each "quadrant multiplication" as a single unit
        # that gets expanded by the half-size decomposition.
        
        # For term t, we have:
        # U_t is formed by summing U_half for each A-quadrant involved, mapped to global indices.
        # V_t is formed by summing V_half for each B-quadrant involved.
        # W_t is formed by summing W_half for each C-quadrant involved.
        
        # Let's define the A and B side combinations for the 7 terms:
        # A_side: list of (br, bc, sign)
        # B_side: list of (br, bc, sign)
        # C_side: list of (br, bc, sign) -- this is where the result goes
        
        # Term 0: S1
        A0 = [(0,0,1), (1,1,1)]
        B0 = [(0,0,1), (1,1,1)]
        C0 = [(0,0,1), (1,1,1)] # S1 contributes to C11 and C22? 
        # Wait. Strassen formulas:
        # C11 = S1 - S4 + S5 - S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S3 + S6 - S7
        
        # So for W, we don't just add to one quadrant. We need to distribute the term's result.
        # The tensor decomposition must satisfy:
        # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k]
        
        # For a single Strassen term t with rank_half decomposition (U_h, V_h, W_h):
        # The term computes M_t = A_t * B_t where A_t and B_t are linear combos of quadrants.
        # Then C is updated by adding appropriate linear combos of M_t to quadrants.
        
        # So for each term t:
        # For each (br_a, bc_a, sign_a) in A_t:
        #   For each local index (li, lj) in U_h:
        #     Global U index (gi, gj) = map(li, lj, br_a, bc_a)
        #     U[gi*N+gj, global_col] += sign_a * U_h[li*N+lj, local_col]
        
        # Similarly for V.
        
        # For W:
        # The result M_t is a (N/2)x(N/2) matrix.
        # It contributes to C quadrants based on the Strassen formula.
        # For each (br_c, bc_c, sign_c) in C_t:
        #   For each local index (li, lj) in W_h:
        #     # W_h indices are (k, i) in local coordinates.
        #     # Global W index is (k_global, i_global).
        #     # li maps to k_local, lj maps to i_local.
        #     # The quadrant (br_c, bc_c) determines where this M_t goes.
        #     # M_t's "row" is k, "col" is i.
        #     # So k_global = br_c * half + li
        #     # i_global = bc_c * half + lj
        #     # W index = k_global * N + i_global
        #     W[k_global*N + i_global, global_col] += sign_c * W_h[li*N+lj, local_col]
        
        # Let's define the C combinations for each term:
        # S1: C11(+), C22(+)  -> [(0,0,1), (1,1,1)]
        # S2: C21(+)          -> [(1,0,1)]
        # S3: C12(+)          -> [(0,1,1)]
        # S4: C11(-), C21(+)  -> [(0,0,-1), (1,0,1)]
        # S5: C11(+), C12(+)  -> [(0,0,1), (0,1,1)]
        # S6: C22(+)          -> [(1,1,1)]
        # S7: C11(-), C22(-)  -> [(0,0,-1), (1,1,-1)]
        
        # A combinations:
        # S1: A11(+), A22(+) -> [(0,0,1), (1,1,1)]
        # S2: A21(+), A22(+) -> [(1,0,1), (1,1,1)]
        # S3: A11(+)         -> [(0,0,1)]
        # S4: A22(+)         -> [(1,1,1)]
        # S5: A11(+), A12(+) -> [(0,0,1), (0,1,1)]
        # S6: A21(+), A11(-) -> [(1,0,1), (0,0,-1)]
        # S7: A12(+), A22(-) -> [(0,1,1), (1,1,-1)]
        
        # B combinations:
        # S1: B11(+), B22(+) -> [(0,0,1), (1,1,1)]
        # S2: B11(+)         -> [(0,0,1)]
        # S3: B12(+), B22(-) -> [(0,1,1), (1,1,-1)]
        # S4: B21(+), B11(-) -> [(1,0,1), (0,0,-1)]
        # S5: B22(+)         -> [(1,1,1)]
        # S6: B11(+), B12(+) -> [(0,0,1), (0,1,1)]
        # S7: B21(+), B22(+) -> [(1,0,1), (1,1,1)]
        
        A_combos = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,1,-1)]
        ]
        B_combos = [
            [(0,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (0,0,-1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (1,1,1)]
        ]
        C_combos = [
            [(0,0,1), (1,1,1)],
            [(1,0,1)],
            [(0,1,1)],
            [(0,0,-1), (1,0,1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (1,1,-1)]
        ]
        
        for t in range(7):
            col_offset = t * rank_half
            for lc in range(rank_half):
                global_col = col_offset + lc
                
                # Process U
                for (br_a, bc_a, sign_a) in A_combos[t]:
                    for li in range(half):
                        for lj in range(half):
                            gi, gj = map_idx(li, lj, br_a, bc_a)
                            val = sign_a * U_half[li*half+lj][lc]
                            U[gi*N+gj][global_col] += val
                            
                # Process V
                for (br_b, bc_b, sign_b) in B_combos[t]:
                    for li in range(half):
                        for lj in range(half):
                            gi, gj = map_idx(li, lj, br_b, bc_b)
                            val = sign_b * V_half[li*half+lj][lc]
                            V[gi*N+gj][global_col] += val
                            
                # Process W
                # W_half indices: (k_loc, i_loc) -> row = k_loc*half + i_loc? 
                # No, W_half is for (N/2)x(N/2) multiplication.
                # Its local indices are (k, i) where k,i in [0, half).
                # W_half index = k * (N/2) + i ? No, W_half is flattened as (k*N/2 + i).
                # In our recursive call, W_half has size (half*half, rank_half).
                # Index is k_loc * half + i_loc.
                
                for (br_c, bc_c, sign_c) in C_combos[t]:
                    for k_loc in range(half):
                        for i_loc in range(half):
                            k_g = br_c * half + k_loc
                            i_g = bc_c * half + i_loc
                            val = sign_c * W_half[k_loc*half+i_loc][lc]
                            W[k_g*N+i_g][global_col] += val
        
        return U, V, W
    
    def get_schoolbook_decomposition(N, P):
        """
        Returns U, V, W for N x N x P matrix multiplication.
        U shape: (N*N, rank)
        V shape: (N*P, rank)
        W shape: (P*N, rank)
        Rank = N*N*P
        """
        rank = N * N * P
        U = [[0] * rank for _ in range(N*N)]
        V = [[0] * rank for _ in range(N*P)]
        W = [[0] * rank for _ in range(P*N)]
        
        col = 0
        for i in range(N):
            for j in range(N):
                for k in range(P):
                    U[i*N+j][col] = 1
                    V[j*P+k][col] = 1
                    W[k*N+i][col] = 1
                    col += 1
        return U, V, W

    # Generate factors for each block
    factors = []
    
    # Block 0: 4x4x4 Strassen
    if time.time() > deadline - 5.0:
        # Fallback to schoolbook if time is tight
        pass
        
    U_s, V_s, W_s = get_strassen_decomposition(4)
    factors.append((U_s, V_s, W_s, 4, 0)) # (U, V, W, p_block_size, k_start)
    
    # Block 1: 4x4x4 Strassen
    if time.time() > deadline - 5.0:
        pass
    factors.append((U_s, V_s, W_s, 4, 4))
    
    # Block 2: 4x4x1 Schoolbook
    U_sc, V_sc, W_sc = get_schoolbook_decomposition(4, 1)
    factors.append((U_sc, V_sc, W_sc, 1, 8))
    
    # Assemble global matrices
    total_rank = sum(f[0][0].__len__() for f in factors)
    U_global = [[0] * total_rank for _ in range(n*m)]
    V_global = [[0] * total_rank for _ in range(m*p)]
    W_global = [[0] * total_rank for _ in range(p*n)]
    
    col_offset = 0
    for U_b, V_b, W_b, p_size, k_start in factors:
        rank_b = len(U_b[0])
        
        # Map U: indices (i, j) are global (i, j) since n, m are same.
        # U_b is (n*m, rank_b). Global U is (n*m, total_rank).
        # Just copy columns.
        for i in range(n*m):
            for c in range(rank_b):
                U_global[i][col_offset + c] = U_b[i][c]
                
        # Map V: 
        # V_b is (m*p_size, rank_b). 
        # Local indices: j*p_size + k_loc.
        # Global indices: j*p + k_global, where k_global = k_start + k_loc.
        for j in range(m):
            for k_loc in range(p_size):
                local_idx = j * p_size + k_loc
                global_idx = j * p + (k_start + k_loc)
                for c in range(rank_b):
                    V_global[global_idx][col_offset + c] = V_b[local_idx][c]
                    
        # Map W:
        # W_b is (p_size*n, rank_b).
        # Local indices: k_loc*n + i.
        # Global indices: k_global*n + i, where k_global = k_start + k_loc.
        for k_loc in range(p_size):
            for i in range(n):
                local_idx = k_loc * n + i
                global_idx = (k_start + k_loc) * n + i
                for c in range(rank_b):
                    W_global[global_idx][col_offset + c] = W_b[local_idx][c]
                    
        col_offset += rank_b
        
    return {"U": U_global, "V": V_global, "W": W_global}
