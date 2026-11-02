import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 3, 6
    
    def make_schoolbook(n, m, p):
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
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_laderman_3x3x3():
        """
        Returns 23 terms for 3x3x3 matrix multiplication.
        Each term is (u, v, w) where u, v, w are 3x3 matrices (flattened to 9 elements).
        The tensor T_{i,j,k} = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] should equal 1.
        """
        # Laderman's 23 terms for C = A * B
        # Standard representation from literature
        terms = []
        
        # Helper to create a 3x3 matrix as a list of 9 elements (row-major)
        def mat3x3(vals):
            return list(vals)
        
        # Term 1: a00*b00 -> c00
        terms.append((
            [1,0,0, 0,0,0, 0,0,0],  # u: a00
            [1,0,0, 0,0,0, 0,0,0],  # v: b00
            [1,0,0, 0,0,0, 0,0,0]   # w: c00
        ))
        
        # Term 2: a00*b01 -> c01
        terms.append((
            [1,0,0, 0,0,0, 0,0,0],  # u: a00
            [0,1,0, 0,0,0, 0,0,0],  # v: b01
            [0,1,0, 0,0,0, 0,0,0]   # w: c01
        ))
        
        # Term 3: a01*b10 -> c00
        terms.append((
            [0,1,0, 0,0,0, 0,0,0],  # u: a01
            [0,0,1, 0,0,0, 0,0,0],  # v: b10
            [1,0,0, 0,0,0, 0,0,0]   # w: c00
        ))
        
        # Term 4: a02*b20 -> c00
        terms.append((
            [0,0,1, 0,0,0, 0,0,0],  # u: a02
            [0,0,0, 0,0,1, 0,0,0],  # v: b20
            [1,0,0, 0,0,0, 0,0,0]   # w: c00
        ))
        
        # Term 5: a10*b01 -> c11
        terms.append((
            [0,0,0, 1,0,0, 0,0,0],  # u: a10
            [0,1,0, 0,0,0, 0,0,0],  # v: b01
            [0,0,0, 1,0,0, 0,0,0]   # w: c11
        ))
        
        # Term 6: a11*b11 -> c11
        terms.append((
            [0,0,0, 0,1,0, 0,0,0],  # u: a11
            [0,0,0, 0,1,0, 0,0,0],  # v: b11
            [0,0,0, 1,0,0, 0,0,0]   # w: c11
        ))
        
        # Term 7: a12*b21 -> c11
        terms.append((
            [0,0,0, 0,0,1, 0,0,0],  # u: a12
            [0,0,0, 0,0,0, 0,1,0],  # v: b21
            [0,0,0, 1,0,0, 0,0,0]   # w: c11
        ))
        
        # Term 8: a20*b02 -> c22
        terms.append((
            [0,0,0, 0,0,0, 1,0,0],  # u: a20
            [0,0,0, 0,0,0, 0,0,1],  # v: b02
            [0,0,0, 0,0,0, 1,0,0]   # w: c22
        ))
        
        # Term 9: a21*b12 -> c22
        terms.append((
            [0,0,0, 0,0,0, 0,1,0],  # u: a21
            [0,0,0, 0,0,0, 0,0,0],  # v: b12
            [0,0,0, 0,0,0, 1,0,0]   # w: c22
        ))
        
        # Term 10: a22*b22 -> c22
        terms.append((
            [0,0,0, 0,0,0, 0,0,1],  # u: a22
            [0,0,0, 0,0,0, 0,0,1],  # v: b22
            [0,0,0, 0,0,0, 1,0,0]   # w: c22
        ))
        
        # Now the non-trivial terms that give the reduction from 27 to 23
        # These involve linear combinations
        
        # Term 11: (a00-a01)*(b01-b10) -> (c01-c00-c11+c10)
        # Actually, let me use the standard Laderman formulation more carefully.
        # The standard 23 terms are:
        # t1: a00*b00 -> c00
        # t2: a01*b10 -> c01  
        # t3: a02*b20 -> c02
        # t4: a10*b01 -> c10
        # t5: a11*b11 -> c11
        # t6: a12*b21 -> c12
        # t7: a20*b02 -> c20
        # t8: a21*b12 -> c21
        # t9: a22*b22 -> c22
        # t10: (a00+a01)*(b10-b11) -> (c01-c00-c11+c10)
        # t11: (a01+a02)*(b20-b21) -> (c02-c01-c12+c11)
        # t12: (a10+a11)*(b01-b02) -> (c10-c11-c21+c20)  -- wait, need to be careful
        # t13: (a11+a12)*(b12-b10) -> ...
        # t14: (a20+a21)*(b02-b00) -> ...
        # t15: (a21+a22)*(b10-b11) -> ...
        # t16: (a00-a11)*(b11-b22) -> ...
        # t17: (a02-a12)*(b10-b00) -> ...
        # t18: (a10-a20)*(b21-b11) -> ...
        # t19: (a00+a22)*(b22-b11) -> ...
        # t20: (a02+a20)*(b11-b22) -> ...
        # t21: (a10+a12)*(b02-b22) -> ...
        # t22: (a01+a21)*(b12-b02) -> ...
        # t23: (a11+a20)*(b20-b01) -> ...
        
        # This is getting complex. Let me use a verified set of coefficients.
        # I'll use the standard Laderman terms as typically presented.
        
        # Let me restart with a cleaner approach using known correct terms.
        # I'll define them as triples of 3x3 matrices.
        
        terms = []
        
        # I'll use the formulation where each term is (U, V, W) 3x3 matrices
        # such that sum of U ⊗ V ⊗ W^T (with appropriate index mapping) gives the tensor
        
        # For C = A*B, the tensor entry T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
        # In our indexing: U[i*m+j], V[j*p+k], W[k*n+i]
        
        # Let me use a different strategy: I'll construct the 23 terms using
        # the known structure and verify them programmatically.
        
        # Actually, let me just use the schoolbook fallback since I'm not 100% confident
        # in the Laderman coefficients from memory. The safe answer is 54.
        
        return None
    
    # First, create the safe schoolbook fallback
    U_school, V_school, W_school, rank_school = make_schoolbook(n, m, p)
    
    # Try to construct rank-46 decomposition using Laderman's 23 terms for each 3x3 block
    # The 3x3x6 tensor = direct sum of two 3x3x3 tensors
    # If we have 23 terms for 3x3x3, we can use them twice (for the two halves of B)
    
    # Let me try to implement Laderman properly
    # Using the standard 23 terms from the literature
    
    def laderman_23_terms():
        """
        Returns 23 triples (u, v, w) where each is a 3x3 matrix as 9-element list (row-major).
        These satisfy: for all i,j,k in {0,1,2}:
        sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
        """
        terms = []
        
        # I'll use the well-known Laderman algorithm terms
        # Reference: Standard matrix multiplication algorithm tables
        
        # The 23 terms (using standard notation where matrices are 3x3):
        # I'll define them as lists of (row, col) positions with coefficients
        
        def make_mat(coefs):
            """Create 3x3 matrix from list of (i,j,val) tuples"""
            mat = [0]*9
            for i, j, val in coefs:
                mat[i*3+j] = val
            return mat
        
        # Term 1: a00*b00 -> c00
        terms.append((
            make_mat([(0,0,1)]),
            make_mat([(0,0,1)]),
            make_mat([(0,0,1)])
        ))
        
        # Term 2: a01*b10 -> c01
        terms.append((
            make_mat([(0,1,1)]),
            make_mat([(1,0,1)]),
            make_mat([(1,0,1)])  # w[k,i] where k=1,i=0 -> index 3
        ))
        
        # Term 3: a02*b20 -> c02
        terms.append((
            make_mat([(0,2,1)]),
            make_mat([(2,0,1)]),
            make_mat([(2,0,1)])  # w[k,i] where k=2,i=0 -> index 6
        ))
        
        # Term 4: a10*b01 -> c10
        terms.append((
            make_mat([(1,0,1)]),
            make_mat([(0,1,1)]),
            make_mat([(0,1,1)])  # w[k,i] where k=0,i=1 -> index 1
        ))
        
        # Term 5: a11*b11 -> c11
        terms.append((
            make_mat([(1,1,1)]),
            make_mat([(1,1,1)]),
            make_mat([(1,1,1)])  # w[k,i] where k=1,i=1 -> index 4
        ))
        
        # Term 6: a12*b21 -> c12
        terms.append((
            make_mat([(1,2,1)]),
            make_mat([(2,1,1)]),
            make_mat([(2,1,1)])  # w[k,i] where k=2,i=1 -> index 7
        ))
        
        # Term 7: a20*b02 -> c20
        terms.append((
            make_mat([(2,0,1)]),
            make_mat([(0,2,1)]),
            make_mat([(0,2,1)])  # w[k,i] where k=0,i=2 -> index 2
        ))
        
        # Term 8: a21*b12 -> c21
        terms.append((
            make_mat([(2,1,1)]),
            make_mat([(1,2,1)]),
            make_mat([(1,2,1)])  # w[k,i] where k=1,i=2 -> index 5
        ))
        
        # Term 9: a22*b22 -> c22
        terms.append((
            make_mat([(2,2,1)]),
            make_mat([(2,2,1)]),
            make_mat([(2,2,1)])  # w[k,i] where k=2,i=2 -> index 8
        ))
        
        # Now the 14 additional terms that reduce from 27 to 23
        # These are the non-trivial bilinear combinations
        
        # Term 10: (a00 - a01) * (b10 - b11) -> (c01 - c00 - c11 + c10)
        terms.append((
            make_mat([(0,0,1), (0,1,-1)]),
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(1,0,1), (0,0,-1), (1,1,-1), (0,1,1)])
        ))
        
        # Term 11: (a01 - a02) * (b20 - b21) -> (c02 - c01 - c12 + c11)
        terms.append((
            make_mat([(0,1,1), (0,2,-1)]),
            make_mat([(2,0,1), (2,1,-1)]),
            make_mat([(2,0,1), (1,0,-1), (2,1,-1), (1,1,1)])
        ))
        
        # Term 12: (a10 - a11) * (b01 - b02) -> (c10 - c11 - c21 + c20)
        terms.append((
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(0,1,1), (0,2,-1)]),
            make_mat([(0,1,1), (1,1,-1), (1,2,-1), (0,2,1)])
        ))
        
        # Term 13: (a11 - a12) * (b12 - b10) -> (c12 - c11 - c22 + c21)
        terms.append((
            make_mat([(1,1,1), (1,2,-1)]),
            make_mat([(1,2,1), (1,0,-1)]),
            make_mat([(2,1,1), (1,1,-1), (2,2,-1), (1,2,1)])
        ))
        
        # Term 14: (a20 - a21) * (b02 - b00) -> (c20 - c21 - c00 + c02)
        terms.append((
            make_mat([(2,0,1), (2,1,-1)]),
            make_mat([(0,2,1), (0,0,-1)]),
            make_mat([(0,2,1), (1,2,-1), (0,0,-1), (2,0,1)])
        ))
        
        # Term 15: (a21 - a22) * (b10 - b11) -> (c21 - c22 - c01 + c02)
        terms.append((
            make_mat([(2,1,1), (2,2,-1)]),
            make_mat([(1,0,1), (1,1,-1)]),
            make_mat([(1,2,1), (2,2,-1), (1,0,-1), (2,0,1)])
        ))
        
        # Term 16: (a00 + a22) * (b22 - b11) -> (c22 + c00 - c02 - c20)
        terms.append((
            make_mat([(0,0,1), (2,2,1)]),
            make_mat([(2,2,1), (1,1,-1)]),
            make_mat([(2,2,1), (0,0,1), (2,0,-1), (0,2,-1)])
        ))
        
        # Term 17: (a02 + a20) * (b11 - b22) -> (c02 + c20 - c00 - c22)
        terms.append((
            make_mat([(0,2,1), (2,0,1)]),
            make_mat([(1,1,1), (2,2,-1)]),
            make_mat([(2,0,1), (0,2,1), (0,0,-1), (2,2,-1)])
        ))
        
        # Term 18: (a10 + a21) * (b12 - b01) -> (c12 + c21 - c11 - c20)
        terms.append((
            make_mat([(1,0,1), (2,1,1)]),
            make_mat([(1,2,1), (0,1,-1)]),
            make_mat([(2,1,1), (1,2,1), (1,1,-1), (0,2,-1)])
        ))
        
        # Term 19: (a12 + a20) * (b02 - b11) -> (c12 + c20 - c10 - c22)
        terms.append((
            make_mat([(1,2,1), (2,0,1)]),
            make_mat([(0,2,1), (1,1,-1)]),
            make_mat([(2,1,1), (0,2,1), (0,1,-1), (2,2,-1)])
        ))
        
        # Term 20: (a01 + a12) * (b21 - b02) -> (c02 + c12 - c01 - c22)
        terms.append((
            make_mat([(0,1,1), (1,2,1)]),
            make_mat([(2,1,1), (0,2,-1)]),
            make_mat([(2,0,1), (2,1,1), (1,0,-1), (2,2,-1)])
        ))
        
        # Term 21: (a10 + a01) * (b11 - b20) -> (c11 + c01 - c10 - c20)
        terms.append((
            make_mat([(1,0,1), (0,1,1)]),
            make_mat([(1,1,1), (2,0,-1)]),
            make_mat([(1,1,1), (1,0,1), (0,1,-1), (0,2,-1)])
        ))
        
        # Term 22: (a20 + a11) * (b02 - b10) -> (c20 + c11 - c22 - c10)
        terms.append((
            make_mat([(2,0,1), (1,1,1)]),
            make_mat([(0,2,1), (1,0,-1)]),
            make_mat([(0,2,1), (1,1,1), (2,2,-1), (0,1,-1)])
        ))
        
        # Term 23: (a21 + a02) * (b22 - b00) -> (c21 + c02 - c22 - c00)
        terms.append((
            make_mat([(2,1,1), (0,2,1)]),
            make_mat([(2,2,1), (0,0,-1)]),
            make_mat([(1,2,1), (2,0,1), (2,2,-1), (0,0,-1)])
        ))
        
        return terms
    
    # Try to build rank-46 decomposition
    def try_laderman_46():
        """
        Attempt to build a rank-46 decomposition for 3x3x6 using
        Laderman's 23 terms for each of the two 3x3 blocks.
        """
        laderman_terms = laderman_23_terms()
        if laderman_terms is None:
            return None
        
        rank = 46
        U = [[0] * rank for _ in range(n * m)]  # 9 x 46
        V = [[0] * rank for _ in range(m * p)]  # 18 x 46
        W = [[0] * rank for _ in range(p * n)]  # 18 x 46
        
        # For the first 3 columns of B (k=0,1,2), use Laderman terms 0-22
        # For the next 3 columns of B (k=3,4,5), use Laderman terms 0-22 again (shifted)
        
        for block in range(2):
            for t in range(23):
                col = block * 23 + t
                u_mat, v_mat, w_mat = laderman_terms[t]
                
                # Fill U: u_mat[i,j] -> U[i*3+j, col]
                for i in range(3):
                    for j in range(3):
                        U[i*3+j][col] = u_mat[i*3+j]
                
                # Fill V: v_mat[j,k] -> V[j*6+(block*3+k), col]
                for j in range(3):
                    for k in range(3):
                        V[j*6 + block*3 + k][col] = v_mat[j*3+k]
                
                # Fill W: w_mat[k,i] -> W[(block*3+k)*3+i, col]
                for k in range(3):
                    for i in range(3):
                        W[(block*3+k)*3+i][col] = w_mat[k*3+i]
        
        return U, V, W, rank
    
    # Try the Laderman approach
    result_46 = try_laderman_46()
    
    if result_46 is not None:
        U, V, W, rank = result_46
        # Verify
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = U_school, V_school, W_school, rank_school
    
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
        
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
