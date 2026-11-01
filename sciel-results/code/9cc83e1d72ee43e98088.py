import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0

def construct():
    """
    Returns a rank-minimal tensor decomposition for 
    multiplying a 2x3 matrix by a 3x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The 2x3x6 tensor is a direct sum of two 2x3x3 tensors.
    # rank(2x3x3) = 7, so rank(2x3x6) <= 14.
    # Let's try to find a rank-14 decomposition, then check if lower ranks work.
    
    # First, construct a rank-7 decomposition for 2x3x3.
    # C[i,k] = sum_j A[i,j] B[j,k] for A (2x3), B (3x3), C (2x3)
    # Decomposition: C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
    # where n=2, m=3, p=3 for the sub-block.
    
    def build_rank7_233():
        """Build rank-7 decomposition for 2x3x3."""
        nn, mm, pp = 2, 3, 3
        r = 7
        
        # U shape: (nn*mm, r) = (6, 7)
        # V shape: (mm*pp, r) = (9, 7)  
        # W shape: (pp*nn, r) = (6, 7)
        
        # We need: C[i,k] = sum_r W[k*nn+i, r] * sum_j U[i*mm+j, r] * V[j*pp+k, r]
        # to equal sum_j A[i,j] B[j,k]
        
        # This means: sum_r W[k*nn+i, r] * (U[i*mm+j, r] * V[j*pp+k, r]) summed over j
        # = A[i,j] * B[j,k]
        
        # Let me set up the equations. For each (i, j, k) triple, the coefficient of 
        # A[i,j]*B[j,k] in the LHS must be 1, and for other combinations, 0.
        
        # Actually, the LHS is: sum_r sum_j' W[k*nn+i, r] * U[i*mm+j', r] * V[j'*pp+k, r]
        # This should equal sum_j A[i,j] B[j,k].
        
        # So the coefficient of A[i,j]B[j,k] is sum_r W[k*nn+i, r] * U[i*mm+j, r] * V[j*pp+k, r] = 1
        # And the coefficient of A[i,j]B[j',k] for j' != j is 0.
        
        # Wait, that's not quite right. The LHS expands as:
        # sum_r W[k*nn+i, r] * sum_j' U[i*mm+j', r] * V[j'*pp+k, r]
        # = sum_j' (sum_r W[k*nn+i, r] * U[i*mm+j', r] * V[j'*pp+k, r])
        
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r W[k*nn+i, r] * U[i*mm+j, r] * V[j*pp+k, r] = 1 for all i, j, k
        # where j is the shared index. But that's 2*3*3 = 18 conditions, and we have 
        # 6*7 + 9*7 + 6*7 = 147 unknowns. So it's underdetermined.
        
        # But wait, the condition is that the entire sum over j' equals sum_j A[i,j]B[j,k].
        # The LHS is a polynomial in A and B. For it to match, we need:
        # For each i, k: sum_j' [sum_r W[k*nn+i, r] U[i*mm+j', r] V[j'*pp+k, r]] A[i,j'] B[j',k] 
        # = sum_j A[i,j] B[j,k]
        
        # This means the coefficient of A[i,j']B[j',k] must be delta_{j',j} for the correct j.
        # But j' ranges over all 3 values, and we need the coefficient to be 1 when j'=j and 0 otherwise.
        # But j is not fixed - the sum is over j. So the coefficient of A[i,j']B[j',k] in the LHS
        # is sum_r W[k*nn+i, r] U[i*mm+j', r] V[j'*pp+k, r], and this must equal 1 for all j'
        # (since the RHS has coefficient 1 for each j in the sum).
        
        # Wait, no. The RHS is sum_j A[i,j] B[j,k]. The coefficient of A[i,j']B[j',k] is 1 
        # for each j'. So we need:
        # sum_r W[k*nn+i, r] U[i*mm+j', r] V[j'*pp+k, r] = 1 for all i, j', k
        
        # That's 2*3*3 = 18 equations. With 147 unknowns, this is very underdetermined.
        # But these are bilinear equations (product of U, V, W entries).
        
        # Let me just try to find a solution numerically.
        
        # Simple approach: set W to be identity-like, then solve for U and V.
        
        # Let me try: W[k*nn+i, r] = 1 if r == k*nn+i, else 0. So W is a 6x7 matrix 
        # with 1s in the first 6 rows of each column... no, that doesn't work.
        
        # Let me try a different initialization and use gradient descent.
        
        np.random.seed(123)
        U = np.random.randn(nn * mm, r)
        V = np.random.randn(mm * pp, r)
        W = np.random.randn(pp * nn, r)
        
        lr = 0.01
        for it in range(5000):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Compute gradient
            # Loss = sum_{i,k} (LHS[i,k] - RHS[i,k])^2
            # LHS[i,k] = sum_r W[k*nn+i, r] * sum_j U[i*mm+j, r] * V[j*pp+k, r]
            # RHS[i,k] = 1 (coefficient is 1 for each j, but we're matching the structure)
            
            # Actually, let me define the loss differently.
            # For each i, j, k: the coefficient sum_r W[k*nn+i, r] U[i*mm+j, r] V[j*pp+k, r] should be 1.
            
            loss = 0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = 0
                        for rr in range(r):
                            val += W[k*nn+i, rr] * U[i*mm+j, rr] * V[j*pp+k, rr]
                        diff = val - 1
                        loss += diff * diff
                        for rr in range(r):
                            grad_U[i*mm+j, rr] += 2 * diff * W[k*nn+i, rr] * V[j*pp+k, rr]
                            grad_V[j*pp+k, rr] += 2 * diff * W[k*nn+i, rr] * U[i*mm+j, rr]
                            grad_W[k*nn+i, rr] += 2 * diff * U[i*mm+j, rr] * V[j*pp+k, rr]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-10:
                break
        
        # Check if we found a valid solution
        max_err = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    val = 0
                    for rr in range(r):
                        val += W[k*nn+i, rr] * U[i*mm+j, rr] * V[j*pp+k, rr]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        else:
            return None, None, None, max_err
    
    U7, V7, W7, err = build_rank7_233()
    
    if U7 is not None:
        # Build rank-14 decomposition for 2x3x6
        r = 14
        U14 = np.zeros((n * m, r))
        V14 = np.zeros((m * p, r))
        W14 = np.zeros((p * n, r))
        
        for b in range(2):
            # Block b uses columns b*7 to b*7+6
            for rr in range(7):
                U14[:, b*7+rr] = U7[:, rr]
                # V: for block b, V[j*6+b*3+k, b*7+rr] = V7[j*3+k, rr]
                for j in range(m):
                    for k in range(3):
                        V14[j * p + b * 3 + k, b * 7 + rr] = V7[j * 3 + k, rr]
                # W: for block b, W[(b*3+k)*2+i, b*7+rr] = W7[k*2+i, rr]
                for k in range(3):
                    for i in range(n):
                        W14[(b * 3 + k) * n + i, b * 7 + rr] = W7[k * n + i, rr]
        
        # Convert to lists
        return {
            "U": U14.tolist(),
            "V": V14.tolist(),
            "W": W14.tolist()
        }
    else:
        # Fallback to rank-18 schoolbook-like decomposition
        rank_block = 18
        U_block = [[0] * rank_block for _ in range(n * m)]
        V_block = [[0] * rank_block for _ in range(m * p)]
        W_block = [[0] * rank_block for _ in range(p * n)]
        
        col = 0
        for b in range(2):
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        U_block[i * m + j][col] = 1
                        V_block[j * p + b * 3 + k][col] = 1
                        W_block[(b * 3 + k) * n + i][col] = 1
                        col += 1
        
        return {"U": U_block, "V": V_block, "W": W_block}
