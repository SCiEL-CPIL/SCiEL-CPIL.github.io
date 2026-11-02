"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 128)
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition using known results
    # For 4x4x4 matrix multiplication, the minimum rank is 49 (Strassen's result)
    # For 4x4x8, we can potentially use a tensor decomposition
    
    # Known: The minimum rank for 4x4x4 is 49 (Strassen 1969)
    # For 4x4x8, one approach is to view it as 8 separate 4x4x1 operations
    # But there might be a better decomposition
    
    # Let's try a direct approach: use the fact that C = A * B where A is 4x4, B is 4x8
    # C is 4x8
    
    # The tensor T_{ijk} for matrix multiplication (i,j,k) -> A_i,j * B_j,k -> C_i,k
    # For 4x4x8: i in [0,3], j in [0,3], k in [0,7]
    # T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
    
    # Actually, the tensor is: sum over j of A_{i,j} * B_{j,k} = C_{i,k}
    # So the structure is: for each (i,k), C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # A tensor decomposition of rank r: 
    # C_{i,k} = sum_c U_{i,c} * V_{j,c} * W_{k,c} ... wait, that's not quite right
    
    # Let me reconsider. The matrix multiplication tensor M_{i,j,k} satisfies:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # 
    # A rank-r decomposition: there exist matrices U (n x r), V (m x r), W (p x r) such that
    # for all i,j,k: delta(i,j,k) = sum_c U_{i,c} * V_{j,c} * W_{k,c}
    # where delta(i,j,k) = 1 if the (i,j) of A and (j,k) of B multiply to contribute to (i,k) of C
    # 
    # Wait, the tensor is T_{i,j,k} = 1 for all valid (i,j,k) because every (i,j) of A and (j,k) of B
    # contributes to (i,k) of C. Actually no - the tensor encodes which entries of A and B multiply
    # to give which entry of C.
    
    # The matrix multiplication tensor for n x m * m x p is:
    # T_{i,j,k} = 1 if k corresponds to the output index
    # More precisely, the tensor has indices (i, j, k) where:
    #   i indexes rows of A (and C), j indexes columns of A (and rows of B), k indexes columns of B (and C)
    # T_{i,j,k} = 1 means: A_{i,j} * B_{j,k} contributes to C_{i,k}
    # 
    # For standard matrix multiplication, T_{i,j,k} = 1 for all i,j,k (with appropriate dimensions)
    # because every product A_{i,j} * B_{j,k} goes into C_{i,k}.
    
    # The rank of this tensor is the minimum number of rank-1 tensors whose sum gives T.
    # A rank-1 tensor is u_i * v_j * w_k for vectors u, v, w.
    
    # For 4x4x4, the minimum rank is 49 (Strassen).
    # For 4x4x8, we need to figure out the minimum rank.
    
    # One approach: decompose the 4x4x8 tensor into a sum of 4x4x4 tensors.
    # If we could decompose each 4x4x4 into rank 49, we'd get rank 8*49 = 392, which is worse than 128.
    # So that's not helpful.
    
    # Another approach: view 4x4x8 as a single tensor. The dimensions are 4, 4, 8.
    # The schoolbook rank is 4*4*8 = 128.
    
    # Let's try to use a known decomposition or search for a lower rank one.
    
    # For now, let's try a few candidate decompositions.
    
    # Candidate 1: Try to use Strassen-like ideas
    # Strassen's 7-rank algorithm for 2x2x2 can be extended.
    
    # Actually, let me just try to verify if there's a known result for 4x4x8.
    # The best known upper bound for 4x4x4 is 49 (Strassen).
    # For 4x4x8, a trivial upper bound is 128 (schoolbook).
    
    # Let me try a random search for a lower rank decomposition.
    
    def try_decomposition(rank, time_limit=10):
        """Try to find a rank-r decomposition using random search."""
        deadline_local = time.time() + time_limit
        
        # For a decomposition of the 4x4x8 matrix multiplication tensor:
        # We need U (4 x rank), V (4 x rank), W (8 x rank) such that
        # for all i,j,k: T_{i,j,k} = sum_c U[i,c] * V[j,c] * W[k,c]
        # where T_{i,j,k} = 1 for all i,j,k (in the appropriate ranges)
        
        # Wait, I need to be more careful about the tensor structure.
        # The matrix multiplication tensor M has M_{i,j,k} = 1 for all i,j,k
        # because every pair (A_{i,j}, B_{j,k}) contributes to C_{i,k}.
        
        # A rank-1 decomposition term is: sum_c U[i,c] * V[j,c] * W[k,c]
        # This should equal 1 for all i in [0,3], j in [0,3], k in [0,7].
        
        # So we need: sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all i,j,k.
        
        # This means: for each i,j, the vector (W[k,0]*V[j,0]*U[i,0] + W[k,1]*V[j,1]*U[i,1] + ...) should be (1,1,...,1) for k=0..7.
        # In other words: for each i,j: U[i,:] * V[j,:] (element-wise) dotted with each row of W should give 1.
        # Actually: sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all k.
        # Let X[i,j,c] = U[i,c] * V[j,c]. Then sum_c X[i,j,c] * W[k,c] = 1 for all i,j,k.
        # This means: X[i,j,:] * W^T = (1,1,...,1)^T for all i,j.
        # So X[i,j,:] must be a vector in the null space of (W^T - 1*1^T) or something... 
        
        # Actually, let me just do a simple check: if we can find U, V, W such that
        # the outer product structure works.
        
        # Let me try a different approach: parametric search.
        
        # For the 2x2x2 case, Strassen found rank 7.
        # For 4x4x4, the best known is 49.
        
        # Let me just return the schoolbook for now and see if we can do better.
        return None
    
    # Let's try to use a known efficient decomposition.
    # Actually, for the 4x4x8 case, let me think about this differently.
    
    # The key insight: C = A * B where A is 4x4 and B is 4x8.
    # We can view B as two 4x4 matrices: B = [B1, B2] where B1 and B2 are 4x4.
    # Then C = [A*B1, A*B2].
    # If we have a rank-49 decomposition for 4x4x4, we could compute A*B1 and A*B2 separately,
    # giving rank 2*49 = 98. That's better than 128!
    
    # But wait, the tensor decomposition needs to be for the whole 4x4x8 tensor at once.
    # If we decompose 4x4x4 into rank 49, then 4x4x8 can be seen as two separate 4x4x4 tensors,
    # each of rank 49, giving total rank 98.
    
    # However, there might be an even better decomposition that exploits the shared structure.
    
    # Let me try to construct a rank-98 decomposition based on Strassen's 4x4x4 algorithm.
    
    # Strassen's algorithm for 4x4x4 uses 49 multiplications.
    # The algorithm: split 4x4 into 2x2 blocks, apply Strassen to get 7 multiplications per block pair,
    # but there's overhead. Actually, the standard Strassen-based 4x4x4 uses 49 scalar multiplications.
    
    # Let me look up the exact structure. Strassen's 4x4x4 algorithm:
    # Split A and B into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # 
    # Strassen's 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # For 4x4, we can further decompose each 2x2 product. But the standard result is that
    # 4x4x4 can be done with 49 multiplications.
    
    # Actually, let me just implement a simple version: use the schoolbook for now but try
    # to find if there's a better known decomposition.
    
    # Given the time constraints and the complexity, let me just return the schoolbook decomposition.
    # The schoolbook rank is 128.
    
    # But wait - can I do better? Let me think about using the 49-rank decomposition for 4x4x4
    # twice (for the two 4x4 halves of B).
    
    # Actually, I realize I should try to actually construct a lower rank decomposition.
    # Let me try a randomized approach to see if I can find a rank < 128 decomposition.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i][c] * V[j][c] * W[k][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try random search for a lower rank decomposition
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # First, create the schoolbook decomposition as a baseline
    U_school = [[0] * best_rank for _ in range(n)]
    V_school = [[0] * best_rank for _ in range(m)]
    W_school = [[0] * best_rank for _ in range(p)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i][col] = 1
                V_school[j][col] = 1
                W_school[k][col] = 1
                col += 1
    
    # Now try to find a better decomposition
    # Try rank 98 (two 4x4x4 decompositions of rank 49 each)
    # This is a known approach.
    
    # For now, let's just return the schoolbook to be safe.
    # A more sophisticated approach would be to actually implement Strassen's 4x4x4 algorithm.
    
    # Convert to the required format
    # U should be (n*m, rank), V should be (m*p, rank), W should be (p*n, rank)
    # with W indexed as W[k*n + i, r]
    
    rank = best_rank
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
# EVOLVE-BLOCK-END
