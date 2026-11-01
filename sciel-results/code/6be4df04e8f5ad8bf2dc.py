"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 8
    pn = p * n  # 8
    
    # Fallback: naive rank-16 decomposition
    def naive_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = naive_decomp()
    
    def check_decomposition(U, V, W, R):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(R))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using integer/half-integer entries
    # The matrix multiplication tensor: T[i,j,k] contributes to C[i,k] = sum_j A[i,j]*B[j,k]
    # For rank r, we need: for all (i,k), sum_{r'} U[i*m+j_r', r'] * V[j_r'*p+k, r'] = delta (only valid j)
    # More precisely: sum_{r'} U[i*m + j, r'] * V[j*p + k, r'] * W[k*n + i, r'] should reconstruct the tensor
    
    # Actually, the condition is: for each (i,j,k) with j in 0..m-1,
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 if this is a valid triple
    # But that's not quite right. The tensor decomposition means:
    # T_{i,j,k} = sum_{r} U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    # where T_{i,j,k} = 1 for all valid (i,j,k)
    
    # Wait, that's not the standard formulation. In standard matrix multiplication:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 for all i,j,k, and the bilinear form is:
    # sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = sum_{i,k} C_{i,k}
    # For a rank-1 term: A_{i,j} B_{j,k} W_{i,k}^{(r)}
    # We need: sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} = 1 for all (i,j,k)
    
    # So the condition is: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of nonlinear equations. Let's try rank 8 first.
    
    def try_rank(R, max_iters=5000, seed=42):
        """Try to find a rank-R decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values in [-1, 1]
        U = np.random.uniform(-1, 1, (nm, R))
        V = np.random.uniform(-1, 1, (mp, R))
        W = np.random.uniform(-1, 1, (pn, R))
        
        # The objective: minimize sum over (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] - 1)^2
        # This is a tensor rank minimization problem
        
        # Use simple gradient descent or coordinate descent
        lr = 0.01
        best_loss = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        residuals[i,j,k] = val - 1.0
            
            loss = np.sum(residuals**2)
            if loss < best_loss:
                best_loss = loss
                best_state = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                break
            
            # Gradient descent step
            # dL/dU[i,j,r] = 2 * residual[i,j,k] * V[j,k,r] * W[i,k,r] (summed over k)
            # Actually for fixed (i,j), the residual depends on k, so:
            # dL/dU[i*m+j, r] = 2 * sum_k residual[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        grad_u = 0.0
                        for k in range(p):
                            grad_u += residuals[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        dU[i*m+j, r] = 2 * grad_u
                        
                        for k in range(p):
                            idx_v = j*p + k
                            idx_w = k*n + i
                            dV[idx_v, r] += 2 * residuals[i,j,k] * U[i*m+j, r] * W[idx_w, r]
                            dW[idx_w, r] += 2 * residuals[i,j,k] * U[i*m+j, r] * V[idx_v, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check best state
        if best_loss < 1e-10:
            U_best, V_best, W_best = best_state
            # Round to nearest half-integer
            U_rounded = np.round(U_best * 2) / 2
            V_rounded = np.round(V_best * 2) / 2
            W_rounded = np.round(W_best * 2) / 2
            
            # Verify
            if check_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # Also try integer rounding
            U_int = np.round(U_best)
            V_int = np.round(V_best)
            W_int = np.round(W_best)
            if check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), R):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    # Try ranks from 8 up to 15
    for R in range(8, min(16, n*m*p)):
        if time.time() > deadline:
            break
        
        result = try_rank(R, max_iters=3000, seed=R)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            break  # Found lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
