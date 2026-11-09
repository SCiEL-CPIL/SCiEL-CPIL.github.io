"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135  # 15s safety margin before 150s kill
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40

    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(m * p)] # Wait, W is p*n
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
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def verify_numpy(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reshape for tensor contraction
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        
        # Compute T_ijk = sum_r U_ir V_jr W_kr
        # Using einsum: 'ijr,jkr,ikr->ijk'
        T = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def try_rank(rank, time_budget):
        """Try to find a rank decomposition with given rank using vectorized gradient descent."""
        start = time.time()
        np.random.seed(42)
        
        # We will try a few random initializations
        for attempt in range(20):
            if time.time() > deadline or time.time() - start > time_budget:
                break
            
            # Random initialization
            # Scale matters. Let's start with small random values.
            U_cur = np.random.randn(n*m, rank) * 0.1
            V_cur = np.random.randn(m*p, rank) * 0.1
            W_cur = np.random.randn(p*n, rank) * 0.1
            
            # Reshape for easier calculation
            U_r = U_cur.reshape(n, m, rank)
            V_r = V_cur.reshape(m, p, rank)
            W_r = W_cur.reshape(p, n, rank)
            
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U_cur)
            vV = np.zeros_like(V_cur)
            vW = np.zeros_like(W_cur)
            
            best_err = np.inf
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute T and Error
                T = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r)
                E = T - 1.0
                err = np.sum(E**2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-12:
                        break
                
                # Compute Gradients
                # dL/dU[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
                # dL/dV[j,k,r] = sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
                # dL/dW[k,i,r] = sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
                
                # Using einsum
                # U_grad: 'ijk,jkr,ikr->ijr'
                U_grad = np.einsum('ijk,jkr,ikr->ijr', E, V_r, W_r)
                # V_grad: 'ijk,ijr,ikr->jkr'
                V_grad = np.einsum('ijk,ijr,ikr->jkr', E, U_r, W_r)
                # W_grad: 'ijk,ijr,jkr->ikr'
                W_grad = np.einsum('ijk,ijr,jkr->ikr', E, U_r, V_r)
                
                # Flatten gradients
                U_grad = U_grad.reshape(n*m, rank)
                V_grad = V_grad.reshape(m*p, rank)
                W_grad = W_grad.reshape(p*n, rank)
                
                # Update with Momentum
                vU = momentum * vU - lr * U_grad
                vV = momentum * vV - lr * V_grad
                vW = momentum * vW - lr * W_grad
                
                U_cur += vU
                V_cur += vV
                W_cur += vW
                
                # Update reshaped views
                U_r = U_cur.reshape(n, m, rank)
                V_r = V_cur.reshape(m, p, rank)
                W_r = W_cur.reshape(p, n, rank)
                
                # Periodic Normalization to handle scale invariance
                # Normalize columns to unit norm? Or just check norms.
                if step % 50 == 0:
                    # Scale factors to keep norms balanced
                    normU = np.linalg.norm(U_cur)
                    normV = np.linalg.norm(V_cur)
                    normW = np.linalg.norm(W_cur)
                    
                    if normU > 0 and normV > 0 and normW > 0:
                        # Target: keep product of norms constant? 
                        # Or just prevent explosion/collapse.
                        # If one norm is very small, gradient is small.
                        # Let's try to keep them roughly equal.
                        avg_norm = (normU + normV + normW) / 3.0
                        if avg_norm > 1e-6:
                            U_cur *= (avg_norm / normU) ** 0.5
                            V_cur *= (avg_norm / normV) ** 0.5
                            W_cur *= (avg_norm / normW) ** 0.5
                            
                            U_r = U_cur.reshape(n, m, rank)
                            V_r = V_cur.reshape(m, p, rank)
                            W_r = W_cur.reshape(p, n, rank)
            
            # Check final result
            U_r = U_cur.reshape(n, m, rank)
            V_r = V_cur.reshape(m, p, rank)
            W_r = W_cur.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r)
            
            # We need integer/half-integer entries.
            # If the solution is close to half-integers, we can round.
            # However, GD might converge to a non-rational point.
            # We should check if the converged values are close to half-integers.
            
            # Try rounding to half-integers
            U_rounded = np.round(U_cur * 2) / 2
            V_rounded = np.round(V_cur * 2) / 2
            W_rounded = np.round(W_cur * 2) / 2
            
            # Verify the rounded solution
            U_r_r = U_rounded.reshape(n, m, rank)
            V_r_r = V_rounded.reshape(m, p, rank)
            W_r_r = W_rounded.reshape(p, n, rank)
            T_r = np.einsum('ijr,jkr,ikr->ijk', U_r_r, V_r_r, W_r_r)
            
            if np.allclose(T_r, 1.0, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Also try integer rounding
            U_int = np.round(U_cur)
            V_int = np.round(V_cur)
            W_int = np.round(W_cur)
            
            U_r_i = U_int.reshape(n, m, rank)
            V_r_i = V_int.reshape(m, p, rank)
            W_r_i = W_int.reshape(p, n, rank)
            T_i = np.einsum('ijr,jkr,ikr->ijk', U_r_i, V_r_i, W_r_i)
            
            if np.allclose(T_i, 1.0, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
            
            # If not close to half-integers, we might still have a valid real solution,
            # but the problem requires integer/half-integer.
            # The problem statement says "exactly decompose ... with integer/half-integer entries".
            # So we MUST find a rational solution.
            
            # Re-seed
            np.random.seed(42 + attempt * 7919)
        
        return None

    # Try to find lower rank decompositions
    # Known: M_{2,4,5} rank is between 34 and 40?
    # Let's try a few ranks.
    # If no lower rank is found, return schoolbook.
    
    # The search space for rank < 40 is difficult.
    # Let's prioritize speed to ensure we at least return the schoolbook if we fail.
    
    for rank in range(34, 40):
        if time.time() > deadline:
            break
        time_budget = min(15, deadline - time.time())
        if time_budget < 2:
            break
        result = try_rank(rank, time_budget)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
