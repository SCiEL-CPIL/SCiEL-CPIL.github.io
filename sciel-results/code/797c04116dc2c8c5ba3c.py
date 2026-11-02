import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses iterative search over rank values, solving the bilinear system.
    """
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # The tensor T: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # a in [0,4), b in [0,14), c in [0,14)
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # Build the list of (a,b,c) where T=1
    tensor_ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_ones.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    one_set = set(tensor_ones)
    
    def verify(U, V, W, rank):
        """Verify the tensor decomposition exactly."""
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    expected = 1.0 if (a, b, c) in one_set else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Try to find minimal rank by solving the system
    # For a given rank R, we need to find U (4×R), V (14×R), W (14×R)
    # such that for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Strategy: use the fact that for fixed b, the equation is linear in U*W
    # T[a,b,c] = sum_r (U[a,r]*W[c,r]) * V[b,r]
    # This is: M_b = V_b^T * (U ⊙ W) where M_b[a,c] = T[a,b,c]
    
    # Alternative: alternate between fixing two matrices and solving for the third
    
    # Let's try a direct approach: for small ranks, use a structured search
    
    # First, let's try the known rank-25 solution as a starting point
    # Then try to reduce rank by merging terms
    
    # Rank 25 solution (from current program)
    def make_rank25():
        rank = 25
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        def add_strassen_block(col_start, k_start):
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
            
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
            
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
            
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
            
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
            
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1
            
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1
        
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        col_start = 21
        k = 6
        U[0, col_start+0] += 1; V[7*0+k, col_start+0] += 1; W[2*k+0, col_start+0] += 1
        U[1, col_start+1] += 1; V[7*1+k, col_start+1] += 1; W[2*k+0, col_start+1] += 1
        U[2, col_start+2] += 1; V[7*0+k, col_start+2] += 1; W[2*k+1, col_start+2] += 1
        U[3, col_start+3] += 1; V[7*1+k, col_start+3] += 1; W[2*k+1, col_start+3] += 1
        
        return U, V, W, rank
    
    # Start with rank 25
    U, V, W, best_rank = make_rank25()
    
    # Try to reduce rank by finding linear dependencies
    # Key idea: if we can express one rank-1 term as a linear combination of others, we can remove it
    # But tensor rank reduction is more subtle
    
    # Alternative approach: try different block structures
    # What if we group columns differently?
    
    # Try: Strassen for columns (0,1), (2,3), and then a combined approach for (4,5,6)
    # Or: use a rank-10 or rank-11 decomposition for the last 3 columns
    
    # Let me try: for 2x2 * 2x3, what's the minimal rank?
    # 2x2 * 2x3 has 12 nonzero tensor entries
    # Schoolbook: 12
    # Can we do better?
    
    # For 2x2 * 2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}
    # This is 6 output entries, each a sum of 2 products
    # Total 12 products, but with shared A entries
    
    # Let me try to find a rank-10 solution for 2x2*2x3
    # Then total would be 7+7+10 = 24
    
    # Or better: find a unified approach
    
    # Let me try: treat the whole thing as A (2x2) times B (2x7)
    # C = A @ B
    # This is equivalent to: for each row i of A, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    
    # Bilinear forms that compute all of C:
    # Let me define:
    # M1 = (A00+A11) * (B0,0+B1,1)  -- Strassen style for cols 0,1
    # etc.
    
    # Actually, let me try a completely different approach:
    # Use the fact that C = A @ B can be computed as:
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    # Define bilinear forms:
    # X1 = A[0,0] * B[0,:]  (vector of 7)
    # X2 = A[0,1] * B[1,:]  (vector of 7)
    # X3 = A[1,0] * B[0,:]  (vector of 7)
    # X4 = A[1,1] * B[1,:]  (vector of 7)
    # C[0,:] = X1 + X2
    # C[1,:] = X3 + X4
    # This is rank 4*7 = 28 (schoolbook)
    
    # To reduce rank, we need to combine these.
    # Strassen combines for 2x2*2x2: uses 7 instead of 4*2=8... wait, 2x2*2x2 schoolbook is 4, Strassen is 7? No.
    # 2x2*2x2: C[0,0]=A00*B00+A01*B10, C[0,1]=A00*B01+A01*B11, C[1,0]=A10*B00+A11*B10, C[1,1]=A10*B01+A11*B11
    # Schoolbook rank: 4
    # Strassen rank: 7? No, Strassen is for 2x2*2x2 and has rank 7? That's worse than 4!
    
    # Wait, I'm confusing myself. Strassen's algorithm computes 2x2*2x2 with 7 multiplications, but the "rank" of the tensor is the number of bilinear forms. For 2x2*2x2, the tensor rank is 7 (Strassen proved it's exactly 7).
    
    # But schoolbook for 2x2*2x2 uses 4 bilinear forms: A00*B00, A01*B10, A10*B00, A11*B10... no wait.
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    # Schoolbook: 8 bilinear forms (each product A[i,j]*B[j,k])
    # Strassen: 7 bilinear forms
    # So Strassen saves 1 for 2x2*2x2.
    
    # For 2x2*2x7: schoolbook is 2*2*7 = 28 bilinear forms.
    # Block Strassen: 3 blocks of 2x2*2x2 (rank 7 each) + 1 column of 2x2*2x1 (rank 3) = 7+7+7+3 = 24?
    # Wait, 2x2*2x1: C[0,k]=A00*B0k+A01*B1k, C[1,k]=A10*B0k+A11*B1k
    # This has 4 products. Can we do it in 3?
    # M1=(A00+A11)*(B0k+B1k), M2=A00*B0k, M3=A11*B1k
    # C[0,k] = M2 + (M1 - M2 - M3) + M3 = M1 - M2... let me check:
    # M1 = A00*B0k + A00*B1k + A11*B0k + A11*B1k
    # M2 = A00*B0k
    # M3 = A11*B1k
    # M1 - M2 - M3 = A00*B1k + A11*B0k
    # C[0,k] = A00*B0k + A01*B1k = M2 + A01*B1k
    # C[1,k] = A10*B0k + A11*B1k = A10*B0k + M3
    # Hmm, I still need A01*B1k and A10*B0k. So that's 5 forms, not 3.
    
    # Actually, 2x2*2x1 has tensor rank 4 (it's just 4 independent products with no structure to exploit).
    # Wait, no. Let me think again.
    # The tensor for 2x2*2x1: T[a,b,c] where a in {0,1,2,3} (A indices), b in {0,1} (B indices), c in {0,1} (C indices)
    # T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1
    # This is 4 nonzero entries. The rank is 4 (each is a separate rank-1 tensor).
    
    # So block Strassen: 7+7+7+4 = 25. That's what we have.
    
    # To do better, we need cross-block sharing.
    
    # Let me try: what if we use a rank-6 or rank-7 decomposition for 2x2*2x3?
    # 2x2*2x3 has 12 nonzero tensor entries.
    # Schoolbook: 12
    # Can we do it in 10 or 11?
    
    # Let me try to find a rank-10 solution for 2x2*2x3 by solving the system.
    
    # For 2x2*2x3:
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[0,2] = A00*B02 + A01*B12
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    # C[1,2] = A10*B02 + A11*B12
    
    # Bilinear forms (rank R):
    # We need 12 equations (one for each (a,b,c) with T=1) and the rest must be 0.
    
    # Let me try R=10 and solve.
    
    # Actually, let me try a more systematic approach: use the "slice" structure.
    # For each b (B index), T[:,b,:] is a 4x14 matrix (for the full problem) or 4x6 (for 2x2*2x3).
    
    # Let me just try to find lower rank solutions by optimization.
    
    def try_find_decomposition(rank, max_time=20.0):
        """Try to find a rank-R decomposition using alternating minimization."""
        if time.time() - start_time > deadline - max_time:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Alternating minimization: fix two, solve for third
        # For fixed U, W: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For fixed a,c: sum_r V[b,r] * (U[a,r]*W[c,r]) = T[a,b,c]
        # This is linear in V[b,:] for each (a,c) pair.
        
        for iteration in range(100):
            if time.time() - start_time > deadline - max_time:
                return None
            
            # Update V: for each b, solve for V[b,:]
            # T[a,b,c] = sum_r V[b,r] * (U[a,r]*W[c,r])
            # For each b: M_b[a,c] = T[a,b,c] = V_b @ (U*W)^T where (U*W)[r, (a,c)] = U[a,r]*W[c,r]
            # Actually: T[:,b,:] = V[b,:] outer product stuff...
            # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # Let M_b = T[:,b,:] (4x14 matrix)
            # M_b[a,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # This is: M_b = (U * V[b,:]) @ (W * V[b,:])^T ... no
            # M_b[a,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
            # = (U * V[b,:]) @ W^T ... where (U*V[b,:])[a,r] = U[a,r]*V[b,r]
            # So M_b = (U * V[b,:]) @ W.T
            # This is bilinear in V[b,:], not linear.
            
            # Alternative: fix U and W, then for each b:
            # T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])  (outer product)
            # Let P_r = U[:,r] * W[:,r] (element-wise product, shape 4x14)
            # Then T[:,b,:] = sum_r V[b,r] * P_r
            # This IS linear in V[b,:]: T[:,b,:] = sum_r V[b,r] * P_r
            # So: T[:,b,:] = P @ V[b,:]^T where P is (4*14, rank) with P[(a,c),r] = U[a,r]*W[c,r]
            
            # Build P matrix
            P = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                for a in range(n*m):
                    for c in range(p*n):
                        P[a * (p*n) + c, r] = U[a, r] * W[c, r]
            
            # For each b: T[:,b,:] (reshaped) = P @ V[b,:]^T
            # T[:,b,:] is 4x14, reshape to 56-vector
            for b in range(m*p):
                target = np.zeros(n*m * p*n)
                for a in range(n*m):
                    for c in range(p*n):
                        if (a, b, c) in one_set:
                            target[a * (p*n) + c] = 1.0
                # Solve: P @ V[b,:]^T = target
                # Least squares
                sol, residuals, rank_p, sv = np.linalg.lstsq(P, target, rcond=None)
                V[b, :] = sol
            
            # Update U: fix V, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b,c: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            # Q[(b,c),r] = V[b,r]*W[c,r], shape (14*14, rank)
            Q = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                for b in range(m*p):
                    for c in range(p*n):
                        Q[b * (p*n) + c, r] = V[b, r] * W[c, r]
            
            for a in range(n*m):
                target = np.zeros(m*p * p*n)
                for b in range(m*p):
                    for c in range(p*n):
                        if (a, b, c) in one_set:
                            target[b * (p*n) + c] = 1.0
                sol, residuals, rank_q, sv = np.linalg.lstsq(Q, target, rcond=None)
                U[a, :] = sol
            
            # Update W: fix U, V
            # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
            # R[(a,b),r] = U[a,r]*V[b,r], shape (4*14, rank)
            R = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                for a in range(n*m):
                    for b in range(m*p):
                        R[a * (m*p) + b, r] = U[a, r] * V[b, r]
            
            for c in range(p*n):
                target = np.zeros(n*m * m*p)
                for a in range(n*m):
                    for b in range(m*p):
                        if (a, b, c) in one_set:
                            target[a * (m*p) + b] = 1.0
                sol, residuals, rank_r, sv = np.linalg.lstsq(R, target, rcond=None)
                W[c, :] = sol
        
        # Check if we found a valid solution
        if verify(U, V, W, rank):
            return U, V, W, rank
        
        return None
    
    # Try ranks from 24 down
    for rank in range(24, 14, -1):
        if time.time() - start_time > deadline - 30.0:
            break
        
        result = try_find_decomposition(rank, max_time=15.0)
        if result is not None:
            U, V, W, best_rank = result
            break
    
    # If we found a better rank, use it; otherwise use rank 25
    if best_rank < 25:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, _ = make_rank25()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
