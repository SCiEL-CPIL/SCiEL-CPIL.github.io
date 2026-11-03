"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact using Fraction for precision."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True

    def try_rank_reduction(target_rank, max_iters=500):
        """Try to find a decomposition of given rank using gradient descent."""
        if time.time() > deadline:
            return None
            
        # Initialize randomly
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Scale to help convergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        lr = 0.01
        prev_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Compute error: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
            # This is a tensor contraction.
            # We can compute the residual tensor R of shape (n, m, p)
            # R[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
            
            # Efficient computation using einsum
            # U is (nm, r), V is (mp, r), W is (pn, r)
            # We need to reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
            U_reshaped = U.reshape(n, m, target_rank)
            V_reshaped = V.reshape(m, p, target_rank)
            W_reshaped = W.reshape(p, n, target_rank)
            
            # Compute product term
            # P[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is tricky with einsum because indices are coupled.
            # U[i,j,r] * V[j,k,r] * W[k,i,r]
            # We can do:
            # Temp1[i,j,k,r] = U[i,j,r] * V[j,k,r]
            # Then sum over r? No, W depends on k,i.
            
            # Let's just use loops for the residual if rank is small, or einsum carefully.
            # Actually, for n=5, m=5, p=5, rank~100, loops might be slow in Python.
            # Let's use numpy broadcasting.
            
            # U[i,j,r] -> (5, 5, R)
            # V[j,k,r] -> (5, 5, R)
            # W[k,i,r] -> (5, 5, R)
            
            # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # Step 1: U[i,j,r] * V[j,k,r]
            # U is (n, m, r), V is (m, p, r)
            # We need to align j.
            # U[:, j, r] and V[j, :, r]
            # Result[i, j, k, r] = U[i, j, r] * V[j, k, r]
            # Shape (n, m, p, r)
            # This is 5*5*5*100 = 12500 elements. Manageable.
            
            # But memory might be an issue if we do this every iteration.
            # Let's try to compute gradients analytically.
            
            # Gradient of Error w.r.t U[i,j,r]:
            # dE/du = V[j,k,r] * W[k,i,r]
            # Summed over j,k? No, for each i,j,r, the error is sum over k?
            # No, the error is for each (i,j,k).
            # E[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
            # Total Error = sum_{i,j,k} E[i,j,k]^2
            # dE/du[i,j,r] = 2 * sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            
            # This is a sum over k.
            
            # Let's compute the gradient using numpy operations.
            # E is (n, m, p)
            # U_grad[i,j,r] = 2 * sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            
            # Compute E
            # We need P[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # This is a specific contraction.
            # Let's use einsum: 'ijr,jkr,ki r->ijk'? No, W is k,i,r.
            # 'ijr,jkr, kir->ijk'
            # U: (n, m, r)
            # V: (m, p, r)
            # W: (p, n, r)
            
            P = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            E = P - 1.0
            
            # Compute Gradients
            # dU[i,j,r] = 2 * sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            # dV[j,k,r] = 2 * sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
            # dW[k,i,r] = 2 * sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
            
            # Use einsum for gradients
            # dU: 'ijk,jkr,kir->ijr'
            U_grad = 2 * np.einsum('ijk,jkr,kir->ijr', E, V_reshaped, W_reshaped)
            # dV: 'ijk,ijr,kir->jkr'
            V_grad = 2 * np.einsum('ijk,ijr,kir->jkr', E, U_reshaped, W_reshaped)
            # dW: 'ijk,ijr,jkr->kir'
            W_grad = 2 * np.einsum('ijk,ijr,jkr->kir', E, U_reshaped, V_reshaped)
            
            # Update
            U -= lr * U_grad.reshape(n*m, target_rank)
            V -= lr * V_grad.reshape(m*p, target_rank)
            W -= lr * W_grad.reshape(p*n, target_rank)
            
            # Check convergence
            current_error = np.sum(E**2)
            if current_error < 1e-6:
                # Try to round
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded
                U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in U_rounded.tolist()]
                V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in V_rounded.tolist()]
                W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in W_rounded.tolist()]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
            if abs(prev_error - current_error) < 1e-9:
                break
            prev_error = current_error
            
        return None

    best = schoolbook()
    best_rank = n * m * p
    
    if verify(best["U"], best["V"], best["W"]):
        pass
    else:
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    
    # Try to find a lower rank
    # Start with a rank close to schoolbook to make it easier
    for target_rank in [124, 123, 122, 120, 110, 100, 90, 80, 70, 60, 50, 49]:
        if time.time() > deadline:
            break
        result = try_rank_reduction(target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
            
    return best
# EVOLVE-BLOCK-END
