"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def get_rank14_factors():
        """Return the known rank-14 decomposition factors as lists."""
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (Strassen for 2x2 * 2x2 -> top-left 2x2 of output)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[5][0]=1
        W[0][0]=1; W[3][0]=1
        
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        
        U[0][2]=1
        V[1][2]=1; V[5][2]=-1
        W[2][2]=1; W[3][2]=1
        
        U[3][3]=1
        V[4][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        
        U[0][4]=1; U[1][4]=1
        V[5][4]=1
        W[0][4]=-1; W[2][4]=1
        
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        U[1][6]=1; U[3][6]=-1
        V[4][6]=1; V[5][6]=1
        W[0][6]=1
        
        # Right block (Strassen for 2x2 * 2x2 -> bottom-right 2x2 of output)
        U[0][7]=1; U[3][7]=1
        V[2][7]=1; V[7][7]=1
        W[4][7]=1; W[7][7]=1
        
        U[2][8]=1; U[3][8]=1
        V[2][8]=1
        W[5][8]=1; W[7][8]=-1
        
        U[0][9]=1
        V[3][9]=1; V[7][9]=-1
        W[6][9]=1; W[7][9]=1
        
        U[3][10]=1
        V[6][10]=1; V[2][10]=-1
        W[4][10]=1; W[5][10]=1
        
        U[0][11]=1; U[1][11]=1
        V[7][11]=1
        W[4][11]=-1; W[6][11]=1
        
        U[2][12]=1; U[0][12]=-1
        V[2][12]=1; V[3][12]=1
        W[7][12]=1
        
        U[1][13]=1; U[3][13]=-1
        V[6][13]=1; V[7][13]=1
        W[4][13]=1
        
        return U, V, W
    
    def try_rank13_search():
        """
        Attempt to find a rank-13 decomposition using a structured approach.
        The idea: take the rank-14 solution and try to merge two terms by
        adjusting the remaining 12 terms. This is a non-linear system.
        
        We'll use a local search / Newton-like method on the 13-term parameters.
        Initialize by dropping one term from rank-14 and perturbing the rest.
        """
        if time.time() > deadline - 15:
            return None
            
        U14, V14, W14 = get_rank14_factors()
        
        # Convert to numpy for computation
        U14_np = np.array(U14, dtype=float)
        V14_np = np.array(V14, dtype=float)
        W14_np = np.array(W14, dtype=float)
        
        # Build the true multiplication tensor
        T_mult = np.zeros((4, 8, 8))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    T_mult[i*2+j, j*4+k, k*2+i] = 1.0
        
        best_result = None
        best_norm = np.inf
        
        # Try dropping each of the 14 terms and see if we can converge
        for drop_idx in range(14):
            if time.time() > deadline - 5:
                break
                
            # Create initial guess for rank 13 by dropping one column
            U13 = np.delete(U14_np, drop_idx, axis=1).copy()
            V13 = np.delete(V14_np, drop_idx, axis=1).copy()
            W13 = np.delete(W14_np, drop_idx, axis=1).copy()
            
            # Try multiple random perturbations
            for trial in range(3):
                if time.time() > deadline - 5:
                    break
                    
                np.random.seed(100 + drop_idx * 10 + trial)
                U13 = U13 + 0.05 * np.random.randn(4, 13)
                V13 = V13 + 0.05 * np.random.randn(8, 13)
                W13 = W13 + 0.05 * np.random.randn(8, 13)
                
                # Gradient descent with adaptive learning rate
                lr = 0.005
                for step in range(200):
                    if time.time() > deadline - 3:
                        break
                        
                    # Compute current tensor
                    T_curr = np.einsum('ir,jr,kr->ijk', U13, V13, W13)
                    residual = T_curr - T_mult
                    norm = np.linalg.norm(residual)
                    
                    if norm < 1e-8:
                        break
                    
                    # Compute gradients
                    dU = np.einsum('ijk,jr,kr->ir', residual, V13, W13)
                    dV = np.einsum('ijk,ir,kr->jr', residual, U13, W13)
                    dW = np.einsum('ijk,ir,jr->kr', residual, U13, V13)
                    
                    # Update with adaptive learning rate
                    grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                    if grad_norm > 0:
                        effective_lr = lr * min(1.0, 0.1 / grad_norm)
                        U13 -= effective_lr * dU
                        V13 -= effective_lr * dV
                        W13 -= effective_lr * dW
                    
                    # Reduce learning rate slowly
                    if step > 50:
                        lr *= 0.99
            
            # Final check
            T_final = np.einsum('ir,jr,kr->ijk', U13, V13, W13)
            final_norm = np.linalg.norm(T_final - T_mult)
            
            if final_norm < 1e-7 and final_norm < best_norm:
                best_norm = final_norm
                # Convert to lists
                U_list = [[float(U13[i,r]) for r in range(13)] for i in range(4)]
                V_list = [[float(V13[j,r]) for r in range(13)] for j in range(8)]
                W_list = [[float(W13[k,r]) for r in range(13)] for k in range(8)]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def naive_decomp():
        rank = n * m * p  # 16
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
    
    # Try to find rank-13 first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Fall back to rank-14
    U, V, W = get_rank14_factors()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
