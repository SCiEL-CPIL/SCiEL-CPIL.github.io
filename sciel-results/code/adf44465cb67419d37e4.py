"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Construct a rank-11 decomposition for 2x3x3 matrix multiplication
    # Using a known construction based on Strassen-like methods
    # For 2x3x3: U(6, r), V(9, r), W(6, r)
    
    def construct_2x3x3_decomposition(rank=11):
        """
        Construct a rank-11 decomposition for 2x3x3 matrix multiplication.
        We split the 3x3 matrix B into a 2x2 block and a 1x3 row.
        Use Strassen (rank 7) for the 2x2x2 part and handle the remaining terms.
        
        A is 2x3: [[a11, a12, a13], [a21, a22, a23]]
        B is 3x3: [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
        C = A*B is 2x3:
        c11 = a11*b11 + a12*b21 + a13*b31
        c12 = a11*b12 + a12*b22 + a13*b32
        c13 = a11*b13 + a12*b23 + a13*b33
        c21 = a21*b11 + a22*b21 + a23*b31
        c22 = a21*b12 + a22*b22 + a23*b32
        c23 = a21*b13 + a22*b23 + a23*b33
        
        Strategy: Use Strassen for the top-left 2x2 submatrix of B (columns 1,2)
        and handle the third column separately.
        
        Let B = [B1 | b3] where B1 is 3x2 (first two columns) and b3 is 3x1 (third column).
        C = [A*B1 | A*b3]
        
        For A*B1 (2x3 * 3x2 = 2x2):
        Use Strassen decomposition of rank 7.
        
        For A*b3 (2x3 * 3x1 = 2x1):
        This requires 2 rank-1 terms: c13 = a11*b13 + a12*b23 + a13*b33
        and c23 = a21*b13 + a22*b23 + a23*b33
        
        But we need to be careful about sharing. Let's construct explicitly.
        
        Strassen for 2x2x2 (A is 2x2, B is 2x2, C is 2x2):
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b21-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Now for our 2x3x3 problem:
        A = [[a11, a12, a13], [a21, a22, a23]]
        B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
        
        We can view this as:
        C11 = a11*b11 + a12*b21 + a13*b31
        C12 = a11*b12 + a12*b22 + a13*b32
        C13 = a11*b13 + a12*b23 + a13*b33
        C21 = a21*b11 + a22*b21 + a23*b31
        C22 = a21*b12 + a22*b22 + a23*b32
        C23 = a21*b13 + a22*b23 + a23*b33
        
        Let's try a direct construction with 11 terms.
        This is a known result; let me look up or derive the specific coefficients.
        
        Actually, let's use a different approach: we'll use the fact that 
        rank(2x3x3) <= rank(2x2x2) + rank(2x1x3) + rank(1x1x3) = 7 + 2 + 3 = 12.
        But we want 11.
        
        Let me try a more systematic approach. We'll use random search with 
        gradient descent to find a rank-11 decomposition.
        """
        n2, m2, p2 = 2, 3, 3
        U = np.zeros((n2 * m2, rank))
        V = np.zeros((m2 * p2, rank))
        W = np.zeros((p2 * n2, rank))
        
        # Initialize with random values
        np.random.seed(42)
        U[:] = np.random.randn(n2 * m2, rank) * 0.1
        V[:] = np.random.randn(m2 * p2, rank) * 0.1
        W[:] = np.random.randn(p2 * n2, rank) * 0.1
        
        # Use gradient descent to minimize the error
        lr = 0.01
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Compute the reconstructed tensor
            # T[i,j,k] = sum_r U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
            errors = np.zeros((n2, m2, p2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        errors[i,j,k] = sum(U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] for r in range(rank)) - 1.0
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        e = errors[i,j,k]
                        for r in range(rank):
                            grad_U[i*m2+j, r] += e * V[j*p2+k, r] * W[k*n2+i, r]
                            grad_V[j*p2+k, r] += e * U[i*m2+j, r] * W[k*n2+i, r]
                            grad_W[k*n2+i, r] += e * U[i*m2+j, r] * V[j*p2+k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            max_err = np.max(np.abs(errors))
            if max_err < 1e-8:
                break
        
        # Verify
        valid = True
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = sum(U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
        else:
            return None
    
    # Try to construct the 2x3x3 decomposition
    decomp_2x3x3 = construct_2x3x3_decomposition(rank=11)
    
    if decomp_2x3x3 is not None:
        U_2x3x3, V_2x3x3, W_2x3x3 = decomp_2x3x3
        rank_2x3x3 = 11
        
        # Now construct 2x3x6 by using two copies
        # U(6, 22), V(18, 22), W(12, 22)
        R = 2 * rank_2x3x3
        U_2x3x6 = np.zeros((n * m, R))
        V_2x3x6 = np.zeros((m * p, R))
        W_2x3x6 = np.zeros((p * n, R))
        
        # First copy: for columns 0-2 of B (3x3 block)
        # U rows: 0-5 (2x3 = 6 elements)
        # V rows: 0-8 (3x3 = 9 elements)
        # W rows: 0-5 (3x2 = 6 elements, for first 3 columns of C)
        
        U_2x3x6[:, 0:rank_2x3x3] = U_2x3x3
        V_2x3x6[0:rank_2x3x3*0+9, 0:rank_2x3x3] = V_2x3x3  # V is 9x11
        W_2x3x6[0:6, 0:rank_2x3x3] = W_2x3x3  # W is 6x11 (for first 3 columns of C, which are 3*2=6)
        
        # Second copy: for columns 3-5 of B (3x3 block)
        # U rows: 0-5 (same)
        # V rows: 9-17 (3x3 = 9 elements)
        # W rows: 6-11 (3x2 = 6 elements, for last 3 columns of C)
        
        U_2x3x6[:, rank_2x3x3:2*rank_2x3x3] = U_2x3x3
        V_2x3x6[9:18, rank_2x3x3:2*rank_2x3x3] = V_2x3x3
        W_2x3x6[6:12, rank_2x3x3:2*rank_2x3x3] = W_2x3x3
        
        # Verify the full 2x3x6 decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_2x3x6[i*m+j, r] * V_2x3x6[j*p+k, r] * W_2x3x6[k*n+i, r] for r in range(R))
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists
            best_U = U_2x3x6.tolist()
            best_V = V_2x3x6.tolist()
            best_W = W_2x3x6.tolist()
            best_rank = R
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
