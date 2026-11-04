"""
Tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses a structured search to find rank < 54 (schoolbook).
"""
import numpy as np
import itertools
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # The tensor T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k]
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Let's try to find rank 27 or lower.
    
    # Strategy: Try rank from 27 upward, use random initialization + gradient descent
    # or try structured patterns.
    
    # First, let's try a simple structured approach for rank 27.
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 6 x 27 x 18
    
    # For rank 27, we can try:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Let's try a systematic search with small integer/rational entries.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W, rank):
        """Verify with Fraction for exactness."""
        from fractions import Fraction
        Uf = [[Fraction(U[i,j]) for j in range(rank)] for i in range(len(U))]
        Vf = [[Fraction(V[i,j]) for j in range(rank)] for i in range(len(V))]
        Wf = [[Fraction(W[i,j]) for j in range(rank)] for i in range(len(W))]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                    if s != Fraction(1):
                        return False
        return True
    
    # Try rank 27 with a structured pattern
    # Idea: For each (j,k) pair, assign a rank index. There are m*p = 27 such pairs.
    # Then V[j*p+k, r] is nonzero only when r = j*p+k, and V[j*p+k, j*p+k] = 1.
    # Then we need: sum_r U[i*m+j, r] * W[k*n+i, r] = 1 for all i,k
    # with U[i*m+j, r] nonzero only for r in some set depending on j,
    # and W[k*n+i, r] nonzero only for r in some set depending on k.
    
    # If V is identity-like (one hot per row), then:
    # U[i*m+j, j*p+k] * W[k*n+i, j*p+k] summed over k... 
    # Actually if V[j*p+k, r] = delta(r, j*p+k), then:
    # sum_r U[i*m+j,r] * delta(r,j*p+k) * W[k*n+i,r] = U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
    # This must equal 1 for all i,k.
    # So we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i,j,k.
    # This means for each fixed j, the matrices U_j (n x p) and W_j^T (n x p) must satisfy
    # U_j[i,k] * W_j[k,i] = 1 for all i,k.
    # This forces U_j[i,k] = a[i,k] and W_j[k,i] = 1/a[i,k] for nonzero entries.
    # But we also need consistency across different j values for the same (i,k) in W.
    # W[k*n+i, j*p+k] appears for each j. So W[k*n+i, r] where r = j*p+k.
    # For fixed (k,i), as j varies, r varies. So W[k*n+i, r] can be different for each j.
    # Similarly U[i*m+j, r] for fixed (i,k) as j varies.
    
    # Let's try: U[i*m+j, j*p+k] = 1, W[k*n+i, j*p+k] = 1 for all i,j,k.
    # Then the product is 1*1 = 1. Let's check if this works!
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # That didn't work because multiple j's contribute to the same (i,k) in W.
    # Let me reconsider.
    
    # Actually, let's just do a numerical optimization for rank 27.
    
    def optimize_rank(rank, max_iters=5000, tol=1e-10):
        """Try to find a decomposition of given rank using gradient descent."""
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Loss: sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        def loss_and_grad():
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute the product
                        prod = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = prod - 1.0
                        loss += err * err
                        
                        # Gradient: d/dU[i*m+j,r] = 2*err * V[j*p+k,r] * W[k*n+i,r]
                        # d/dV[j*p+k,r] = 2*err * U[i*m+j,r] * W[k*n+i,r]
                        # d/dW[k*n+i,r] = 2*err * U[i*m+j,r] * V[j*p+k,r]
                        
                        dU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                        dV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                        dW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
            
            return loss, dU, dV, dW
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            loss, dU, dV, dW = loss_and_grad()
            if loss < tol:
                return U, V, W
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                loss2, _, _, _ = loss_and_grad()
                if loss2 > loss:
                    lr *= 0.5
                elif loss2 < loss * 0.9:
                    lr *= 1.1
            
            if it % 1000 == 0:
                pass  # Could log
        
        final_loss, _, _, _ = loss_and_grad()
        if final_loss < 1e-8:
            return U, V, W
        return None
    
    # Try rank 27
    result = optimize_rank(27)
    if result is not None:
        U, V, W = result
        if verify(U, V, W, 27):
            # Try to round to nice values
            U_rounded = np.round(U, 6)
            V_rounded = np.round(V, 6)
            W_rounded = np.round(W, 6)
            if verify(U_rounded, V_rounded, W_rounded, 27):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 28
    if time.time() < deadline - 30:
        result = optimize_rank(28)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, 28):
                U_rounded = np.round(U, 6)
                V_rounded = np.round(V, 6)
                W_rounded = np.round(W, 6)
                if verify(U_rounded, V_rounded, W_rounded, 28):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 30
    if time.time() < deadline - 20:
        result = optimize_rank(30)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, 30):
                U_rounded = np.round(U, 6)
                V_rounded = np.round(V, 6)
                W_rounded = np.round(W, 6)
                if verify(U_rounded, V_rounded, W_rounded, 30):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
