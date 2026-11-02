"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin for evaluation
    
    def _verify_decomposition(U, V, W, n, m, p, num_trials=20):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using the decomposition:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # Compute left factors: L[i] = sum_j U[i*m+j, r]*A[i,j]
                L = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        L[i] += U[i*m+j, r] * A[i,j]
                # Compute right factors: R[k] = sum_j V[j*p+k, r]*B[j,k]
                R = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        R[k] += V[j*p+k, r] * B[j,k]
                # Add contribution: W[k*n+i, r] * L[i] * R[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i,k] += W[k*n+i, r] * L[i] * R[k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    def _build_schoolbook(n, m, p):
        """Build the schoolbook decomposition (rank = n*m*p)."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    # Try to find a better decomposition using structured search
    # For 3x3 * 3x8, we can try various approaches
    
    # Approach 1: Try to use a decomposition that exploits the structure
    # We can view this as 3 independent 1x3 * 3x8 multiplications (row-wise)
    # or 8 independent 3x3 * 3x1 multiplications (column-wise)
    
    # Let's try a direct optimization approach with multiple random restarts
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    rng = np.random.RandomState(42)
    
    # Try ranks from low to high
    for rank in range(1, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Multiple random restarts for each rank
        num_restarts = max(1, int(30 * (best_rank / rank)))
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Use Adam-like optimization
            lr = 0.01
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            for it in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Sample random matrices
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_expected = A @ B
                
                # Compute C_computed and gradients
                C_computed = np.zeros((n, p))
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Precompute L and R for each r
                L = np.zeros((n, rank))
                R = np.zeros((p, rank))
                for r in range(rank):
                    for i in range(n):
                        for j in range(m):
                            L[i, r] += U[i*m+j, r] * A[i,j]
                    for k in range(p):
                        for j in range(m):
                            R[k, r] += V[j*p+k, r] * B[j,k]
                
                for i in range(n):
                    for k in range(p):
                        # C[i,k] = sum_r W[k*n+i, r] * L[i,r] * R[k,r]
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i, r] * L[i,r] * R[k,r]
                        C_computed[i,k] = val
                        err = val - C_expected[i,k]
                        
                        # Gradients
                        for r in range(rank):
                            dW[k*n+i, r] += err * L[i,r] * R[k,r]
                            # dL[i,r] = err * W[k*n+i, r] * R[k,r]
                            dL_ir = err * W[k*n+i, r] * R[k,r]
                            # dR[k,r] = err * W[k*n+i, r] * L[i,r]
                            dR_kr = err * W[k*n+i, r] * L[i,r]
                            
                            for j in range(m):
                                dU[i*m+j, r] += dL_ir * A[i,j]
                                dV[j*p+k, r] += dR_kr * B[j,k]
                
                # Update with Adam
                t = it + 1
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * dU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * dV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * dW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Check convergence
                if np.max(np.abs(C_computed - C_expected)) < 1e-10:
                    found = True
                    break
            
            if found:
                if _verify_decomposition(U, V, W, n, m, p, num_trials=10):
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
        if found:
            break
    
    # Round to half-integers if possible
    def _round_to_half(arr):
        """Round values to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    U_rounded = _round_to_half(best_U)
    V_rounded = _round_to_half(best_V)
    W_rounded = _round_to_half(best_W)
    
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, num_trials=10):
        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
