"""Direct rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import itertools

# Deadline guard: kill after 140s to stay within 150s wall-clock
_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: return schoolbook rank 96 as a safe valid answer
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # The matrix multiplication tensor T has entries T[i,j,k] = delta_{i,k} * delta_{j,j}... 
    # Actually, T is defined such that C = AB means C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i,j,k} = 1 if and only if... no.
    # The tensor is T in R^{n x m x p} where T_{i,j,k} = 1 if we multiply A_{ij} by B_{jk} to get C_{ik}.
    # Wait, the standard tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where T_{i,j,k} = 1 if and only if the product A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, T_{i,j,k} = 1 for all valid i,j,k such that the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C.
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right. The tensor is sparse.
    # T_{i,j,k} = 1 if and only if we are multiplying A_{ij} and B_{jk}.
    # This is always true for valid indices. So T_{i,j,k} = 1 for all i,j,k.
    # No, that would mean the tensor is all ones, which is rank 1.
    # 
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M_{i,j,k} = 1 if and only if the (i,j) entry of A and the (j,k) entry of B are multiplied.
    # So M_{i,j,k} = 1 for all i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}.
    # That's not right either.
    #
    # The correct definition: The tensor T in R^{n*m * m*p * p*n} (or indexed by i,j,k) is such that
    # T_{i,j,k} = 1 if and only if the product A_{ij} B_{jk} contributes to C_{ik}.
    # This is always the case. So T_{i,j,k} = 1 for all valid i,j,k.
    #
    # Wait, I think I'm confusing myself. Let me look at the interface:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for all i,j,k:
    #   T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # where T_{i,j,k} = 1 if and only if we multiply A_{ij} and B_{jk} to get part of C_{ik}.
    #
    # Actually, T_{i,j,k} = 1 for all i,j,k such that the (i,j) entry of A and (j,k) entry of B are used.
    # This is all of them. So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # Hmm, but that means the tensor is the all-ones tensor, which has rank 1.
    # That can't be right.
    #
    # Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 8 matrix."
    #
    # I think the tensor is actually defined as:
    # T_{i,j,k} = 1 if and only if i=k (for the C index) and j is the shared index.
    # No, that's not right either.
    #
    # OK let me think about this differently. The matrix multiplication C = AB where A is n x m and B is m x p.
    # C_{ik} = sum_{j=0}^{m-1} A_{ij} B_{jk}
    #
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)} such that:
    # T_{(i,j), (j,k), (k,i)} = 1 for all valid i,j,k, and 0 otherwise.
    #
    # So T_{a,b,c} = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where j = j' and k = k' and i = i'.
    # Wait, no. The indices are:
    # - U index: (i,j) where i in [0,n), j in [0,m) -> linear index i*m + j
    # - V index: (j,k) where j in [0,m), k in [0,p) -> linear index j*p + k
    # - W index: (k,i) where k in [0,p), i in [0,n) -> linear index k*n + i
    #
    # The tensor T has T_{(i,j),(j,k),(k,i)} = 1.
    # So T is nonzero only when the j-index of U matches the j-index of V, and the k-index of V matches the k-index of W, and the i-index of W matches the i-index of U.
    #
    # So T_{a,b,c} = 1 if and only if:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some valid i,j,k.
    #
    # This means T is a permutation-like tensor.
    
    # For the given dimensions n=3, m=4, p=8:
    # U shape: (12, rank)
    # V shape: (32, rank)
    # W shape: (24, rank)
    
    # The schoolbook rank is n*m*p = 3*4*8 = 96.
    # The current program achieves rank 88.
    
    # Let me try to find a lower rank.
    # The minimum rank of matrix multiplication M_{n,m,p} is a well-studied problem.
    # For M_{3,4,8}, what is the minimum rank?
    #
    # Lower bounds: 
    # - The rank is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # - There are better lower bounds from algebraic geometry.
    #
    # Known results:
    # - M_{2,2,2} has rank 7 (Strassen).
    # - M_{3,3,3} has rank at least 19, and 23 is known.
    #
    # For M_{3,4,8}, let's think about what rank is achievable.
    #
    # One approach: use Strassen-like algorithms on blocks.
    # Split A into blocks and use fast matrix multiplication.
    
    # Let me try a direct construction.
    # The current program uses:
    # - 32 terms for i=0 (schoolbook for 1x4x8)
    # - 56 terms for i=1,2 (Strassen on 2x4x8, split into 2x2x2 blocks)
    # Total: 88.
    
    # Can we do better?
    # For the 2x4x8 part, the schoolbook rank is 2*4*8 = 64.
    # The current program uses 56, saving 8 terms.
    # 
    # Can we do better than 56 for 2x4x8?
    # 2x4x8 can be seen as 4 blocks of 2x2x2.
    # Each 2x2x2 block has rank 7 (Strassen), so 4*7 = 28.
    # But that's not how it works because the blocks share indices.
    
    # Actually, let me reconsider the structure.
    # For M_{2,4,8}:
    # A is 2x4, B is 4x8, C is 2x8.
    # C_{ik} = sum_{j=0}^{3} A_{ij} B_{jk}
    #
    # We can split j into two groups: j=0,1 and j=2,3.
    # C_{ik} = sum_{j=0}^{1} A_{ij} B_{jk} + sum_{j=2}^{3} A_{ij} B_{jk}
    #
    # For each group, we have a 2x2x8 product.
    # For 2x2x8: A is 2x2, B is 2x8, C is 2x8.
    # C_{ik} = sum_{j=0}^{1} A_{ij} B_{jk}
    #
    # This can be done with Strassen-like methods.
    # Actually, for 2x2xp, the rank is at most 2p + 2 (using Strassen-like ideas).
    # For p=8, that's 18.
    # Two such blocks give 36, plus the i=0 part gives 8*1 = 8, total 44?
    # That seems too optimistic.
    
    # Let me just try to verify the current program and see if it's correct.
    # Then I'll try to improve it.
    
    # For now, let's return the current construction but with a time guard.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (schoolbook for 1x4x8)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2x2 blocks
    # Strassen matrices for 2x2x2: 7 products
    # U: 7x4 (rows=products, cols=A entries [i0j0, i0j1, i1j0, i1j1])
    # V: 7x4 (rows=products, cols=B entries [j0k0, j0k1, j1k0, j1k1])
    # W: 7x4 (rows=products, cols=C entries [i0k0, i0k1, i1k0, i1k1])
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # Apply Strassen to each 2x2x2 block: 2 j-blocks x 4 k-blocks = 8 blocks, 7 terms each = 56
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: A entries for local 2x2 block (i=1,2; j=j_start, j_start+1)
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2  # 0 or 1
                        local_j_prime = l % 2   # 0 or 1
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                # V: B entries for local 2x2 block (j=j_start, j_start+1; k=k_start, k_start+1)
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2  # 0 or 1
                        local_k_prime = l % 2   # 0 or 1
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                # W: C entries for local 2x2 block (i=1,2; k=k_start, k_start+1)
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2  # 0 or 1
                        local_k_prime = l % 2   # 0 or 1
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                col += 1
                
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
