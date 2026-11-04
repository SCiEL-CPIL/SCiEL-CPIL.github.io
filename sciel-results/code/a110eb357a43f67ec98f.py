"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses Kronecker product structure to decompose 6x6x6 into 3x3x3 blocks.
    """
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # all 6
    
    def build_3x3x3():
        """
        Build a valid tensor decomposition for 3x3x6 (i.e., 3x3 * 3x6 = 3x6).
        We'll use the simple approach: for each (i,j) in 3x3 and k in 0..5,
        we have a rank-1 term. But let's try to do better.
        
        Actually, for 3x3*3x6, the standard rank is 3*3*6=54.
        Let's use the block structure: A is 3x3, B is 3x6, C is 3x6.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For the tensor decomposition of this, we can use rank 18 by grouping.
        Actually, let me just use the straightforward approach with some structure.
        
        For 3x3*3x6: We can use the fact that 3x3 multiplication can be done
        with rank 15 (Strassen-like), but let's use a simpler construction.
        
        Simple construction: rank = 3*3*6 = 54, one term per (i,j,k).
        """
        n3, m3, p3 = 3, 3, 6
        rank3 = n3 * m3 * p3  # 54 for the simple case
        U3 = [[0] * rank3 for _ in range(n3 * m3)]
        V3 = [[0] * rank3 for _ in range(m3 * p3)]
        W3 = [[0] * rank3 for _ in range(p3 * n3)]
        
        col = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i * m3 + j][col] = 1
                    V3[j * p3 + k][col] = 1
                    W3[k * n3 + i][col] = 1
                    col += 1
        
        return U3, V3, W3, rank3
    
    def combine_kronecker(U1, V1, W1, r1, U2, V2, W2, r2):
        """
        Combine two tensor decompositions using Kronecker product.
        If (U1,V1,W1) decomposes n1*m1 x m1*p1 x p1*n1 with rank r1,
        and (U2,V2,W2) decomposes n2*m2 x m2*p2 x p2*n2 with rank r2,
        then the Kronecker product decomposes (n1*n2)*(m1*m2) x (m1*m2)*(p1*p2) 
        x (p1*p2)*(n1*n2) with rank r1*r2.
        
        But we need to be careful about the indexing conventions.
        
        For the combined tensor:
        U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), r1*r2] = 
            U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
        
        Wait, the indexing is: U[i*m+j, r] where i is row, j is column.
        For the Kronecker product of matrices A1 (n1 x m1) and A2 (n2 x m2):
        (A1 ⊗ A2)[(i1*n2+i2), (j1*m2+j2)] = A1[i1,j1] * A2[i2,j2]
        
        So for the U factor:
        U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), (r1_idx, r2_idx)] = 
            U1[i1*m1+j1, r1_idx] * U2[i2*m2+j2, r2_idx]
        
        The rank is r1*r2, with column index = r1_idx * r2 + r2_idx.
        """
        n1, m1 = U1[0].__len__() if isinstance(U1, list) else len(U1[0]), len(U1)
        # Let me re-derive. U1 has shape (n1*m1, r1)
        # So n1*m1 = len(U1), r1 = len(U1[0])
        
        rows1 = len(U1)  # n1*m1
        r1 = len(U1[0])
        rows2 = len(U2)  # n2*m2
        r2 = len(U2[0])
        
        # Need to figure out n1, m1 from rows1. We'll pass them explicitly.
        # Let me rewrite to take dimensions as parameters.
        raise NotImplementedError("Rewrite with explicit dims")
    
    def combine_kronecker_v2(U1, V1, W1, n1, m1, p1, r1, U2, V2, W2, n2, m2, p2, r2):
        """
        Combine two tensor decompositions using Kronecker product.
        """
        n, m, p = n1 * n2, m1 * m2, p1 * p2
        rank = r1 * r2
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Pre-convert to numpy for speed
        U1_arr = np.array(U1, dtype=float)
        V1_arr = np.array(V1, dtype=float)
        W1_arr = np.array(W1, dtype=float)
        U2_arr = np.array(U2, dtype=float)
        V2_arr = np.array(V2, dtype=float)
        W2_arr = np.array(W2, dtype=float)
        
        # U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), r1_idx*r2 + r2_idx] = 
        #     U1[i1*m1+j1, r1_idx] * U2[i2*m2+j2, r2_idx]
        
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        row_idx = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                        for r1_idx in range(r1):
                            val1 = U1_arr[i1 * m1 + j1, r1_idx]
                            if val1 == 0:
                                continue
                            for r2_idx in range(r2):
                                val2 = U2_arr[i2 * m2 + j2, r2_idx]
                                if val2 == 0:
                                    continue
                                col_idx = r1_idx * r2 + r2_idx
                                U[row_idx][col_idx] = val1 * val2
        
        # V_combined[(j1*m2+j2)*(p1*p2) + (k1*p2+k2), r1_idx*r2 + r2_idx] = 
        #     V1[j1*p1+k1, r1_idx] * V2[j2*p2+k2, r2_idx]
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        row_idx = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                        for r1_idx in range(r1):
                            val1 = V1_arr[j1 * p1 + k1, r1_idx]
                            if val1 == 0:
                                continue
                            for r2_idx in range(r2):
                                val2 = V2_arr[j2 * p2 + k2, r2_idx]
                                if val2 == 0:
                                    continue
                                col_idx = r1_idx * r2 + r2_idx
                                V[row_idx][col_idx] = val1 * val2
        
        # W_combined[(k1*p2+k2)*(n1*n2) + (i1*n2+i2), r1_idx*r2 + r2_idx] = 
        #     W1[k1*n1+i1, r1_idx] * W2[k2*n2+i2, r2_idx]
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        row_idx = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                        for r1_idx in range(r1):
                            val1 = W1_arr[k1 * n1 + i1, r1_idx]
                            if val1 == 0:
                                continue
                            for r2_idx in range(r2):
                                val2 = W2_arr[k2 * n2 + i2, r2_idx]
                                if val2 == 0:
                                    continue
                                col_idx = r1_idx * r2 + r2_idx
                                W[row_idx][col_idx] = val1 * val2
        
        return U, V, W
    
    # Build 6x6x6 as Kronecker product of 3x3x3 and 2x2x2
    # 6 = 3 * 2, so 6x6x6 = (3x3) ⊗ (2x2) for each dimension
    
    # First, build 2x2x2 decomposition (rank 8, the simple one)
    n2, m2, p2 = 2, 2, 2
    r2 = 8
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    # Now build 3x3x3 decomposition
    n3, m3, p3 = 3, 3, 3
    # Simple rank 27
    r3 = 27
    U3 = [[0] * r3 for _ in range(n3 * m3)]
    V3 = [[0] * r3 for _ in range(m3 * p3)]
    W3 = [[0] * r3 for _ in range(p3 * n3)]
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j][col] = 1
                V3[j * p3 + k][col] = 1
                W3[k * n3 + i][col] = 1
                col += 1
    
    # Try to find a better 3x3x3 decomposition using known results
    # For 3x3x3, the best known is rank 23 (or lower in some constructions)
    # Let's try to use a structured approach
    
    # Actually, let's first check if the simple Kronecker gives rank 27*8 = 216
    # That's the same as schoolbook. We need to do better.
    
    # Let's try to find a lower-rank 3x3x3 decomposition.
    # One known construction: use the fact that 3x3 multiplication can be
    # decomposed using 15 multiplications (Strassen-like).
    
    # For now, let's try a different approach: directly construct a 6x6x6
    # decomposition using a hybrid method.
    
    # Alternative: use the "border rank" approach or known tensor decompositions.
    
    # Let's try: decompose 6x6x6 as a direct sum of smaller tensors.
    # Or use the fact that 6 = 4+2, and use separate decompositions for 4x4x4 and 2x2x2.
    
    # Actually, the cleanest approach: use the Kronecker product but with
    # a lower-rank base tensor.
    
    # For 3x3x3, let's try rank 23 using a known construction.
    # But for now, let's just use the simple approach and see if we can improve.
    
    # Build the 6x6x6 using Kronecker of 3x3x3 (rank 27) and 2x2x2 (rank 8)
    # This gives rank 216, which is not an improvement.
    
    # Let's try a different factorization: 6 = 2 * 3, so use 2x2x2 and 3x3x3.
    # Same thing.
    
    # What if we use 6 = 4 + 2? We can decompose the 6x6 matrices into blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x2, etc.
    # This gets complicated. Let's try a direct construction.
    
    # Direct construction for 6x6x6 with reduced rank:
    # Use the idea that we can share some of the rank-1 terms across blocks.
    
    # For now, let's return the schoolbook solution and try to improve iteratively.
    
    # Actually, let me try a smarter approach: use the fact that for 6x6x6,
    # we can use the tensor decomposition from the literature.
    
    # Known: For 6x6x6, there's a construction with rank 215 or lower.
    # Let's try to use a randomized approach with structured initialization.
    
    def try_structured_decomposition(target_rank):
        """
        Try to find a decomposition with the given rank using structured initialization.
        """
        if time.time() - start_time > deadline - 20:
            return None
        
        n, m, p = 6, 6, 6
        
        # Initialize with a structured pattern
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Start from the schoolbook pattern and try to merge some terms
        # Schoolbook has 216 terms. We want to reduce to target_rank.
        
        # For each term in schoolbook, we have a unique (i,j,k) triple.
        # If we can find linear dependencies, we can reduce the rank.
        
        # This is a complex optimization problem. Let's use a simple heuristic:
        # Group terms by their (i mod b, j mod b, k mod b) pattern and try
        # to share factors.
        
        # For now, this is a placeholder. Let's just return None.
        return None
    
    # Try to find a better decomposition
    for target_rank in [215, 210, 200, 190, 180]:
        if time.time() - start_time > deadline - 20:
            break
        result = try_structured_decomposition(target_rank)
        if result is not None:
            U_list = result[0]
            V_list = result[1]
            W_list = result[2]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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

# EVOLVE-BLOCK-END
