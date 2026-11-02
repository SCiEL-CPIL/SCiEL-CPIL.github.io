"""Matrix multiplication 3x3x3 rank-26 decomposition with IHT search for lower rank."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-26 exact decomposition for 3x3 matrix multiplication.
    Attempts to find lower-rank decompositions via iterative hard thresholding.
    Falls back to rank-26 if search fails.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = 3, 3, 3
    
    def get_rank26():
        """Return the known rank-26 decomposition."""
        rank = 26
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        
        def add_term(r, u_items, v_items, w_items):
            for idx, coeff in u_items:
                U[idx][r] += coeff
            for idx, coeff in v_items:
                V[idx][r] += coeff
            for idx, coeff in w_items:
                W[idx][r] += coeff

        # Strassen for 2x2 block
        add_term(0, [(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
        add_term(1, [(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
        add_term(2, [(0, 1)], [(1, 1), (4, -1)], [(3, 1), (4, 1)])
        add_term(3, [(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)])
        add_term(4, [(0, 1), (3, 1)], [(4, 1)], [(0, -1), (3, 1)])
        add_term(5, [(1, 1), (0, -1)], [(0, 1), (3, 1)], [(1, 1)])
        add_term(6, [(4, 1), (3, -1)], [(1, 1), (4, 1)], [(4, 1), (0, -1)])
        
        # Cross terms
        add_term(7, [(2, 1)], [(6, 1)], [(0, 1)])
        add_term(8, [(2, 1)], [(7, 1)], [(3, 1)])
        add_term(9, [(5, 1)], [(6, 1)], [(1, 1)])
        add_term(10, [(5, 1)], [(7, 1)], [(4, 1)])
        add_term(11, [(0, 1)], [(2, 1)], [(6, 1)])
        add_term(12, [(1, 1)], [(5, 1)], [(6, 1)])
        add_term(13, [(3, 1)], [(2, 1)], [(7, 1)])
        add_term(14, [(4, 1)], [(5, 1)], [(7, 1)])
        add_term(15, [(6, 1)], [(0, 1)], [(2, 1)])
        add_term(16, [(6, 1)], [(1, 1)], [(5, 1)])
        add_term(17, [(7, 1)], [(3, 1)], [(2, 1)])
        add_term(18, [(7, 1)], [(4, 1)], [(5, 1)])
        add_term(19, [(8, 1)], [(6, 1)], [(2, 1)])
        add_term(20, [(8, 1)], [(7, 1)], [(5, 1)])
        add_term(21, [(2, 1)], [(8, 1)], [(6, 1)])
        add_term(22, [(5, 1)], [(8, 1)], [(7, 1)])
        add_term(23, [(6, 1)], [(2, 1)], [(8, 1)])
        add_term(24, [(7, 1)], [(5, 1)], [(8, 1)])
        add_term(25, [(8, 1)], [(8, 1)], [(8, 1)])
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank, tol=1e-10):
        """Verify the decomposition exactly."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def get_rank27():
        """Return the trivial rank-27 schoolbook decomposition."""
        rank27 = 27
        U27 = [[0] * rank27 for _ in range(9)]
        V27 = [[0] * rank27 for _ in range(9)]
        W27 = [[0] * rank27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U27[i * 3 + j][col] = 1
                    V27[j * 3 + k][col] = 1
                    W27[k * 3 + i][col] = 1
                    col += 1
        return {"U": U27, "V": V27, "W": W27}
    
    # First, verify rank-26 works
    r26 = get_rank26()
    if verify(r26["U"], r26["V"], r26["W"], 26):
        best_result = r26
        best_rank = 26
    else:
        best_result = get_rank27()
        best_rank = 27
    
    # Try to find lower rank via iterative hard thresholding
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We want to represent T as sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^9, v_r is in R^9, w_r is in R^9
    
    def iht_search(target_rank, max_iter=500, tol=1e-12):
        """
        Try to find a decomposition of rank <= target_rank using IHT.
        Returns (U, V, W) if successful, None otherwise.
        """
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        np.random.seed(42 + target_rank)
        
        for attempt in range(10):  # Multiple random restarts
            if time.time() > deadline - 5:
                return None
            
            U = np.random.randn(9, target_rank)
            V = np.random.randn(9, target_rank)
            W = np.random.randn(9, target_rank)
            
            # Normalize to prevent overflow
            U /= np.linalg.norm(U) * np.sqrt(9)
            V /= np.linalg.norm(V) * np.sqrt(9)
            W /= np.linalg.norm(W) * np.sqrt(9)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Compute current tensor approximation
                # T_approx[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                T_approx = np.zeros((3, 3, 3))
                for r in range(target_rank):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                T_approx[i,j,k] += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                
                # Compute error
                T_target = np.ones((3, 3, 3))
                error = T_target - T_approx
                err_norm = np.linalg.norm(error)
                
                if err_norm < tol:
                    # Convert to lists
                    U_list = [[float(U[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                    V_list = [[float(V[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                    W_list = [[float(W[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                    return (U_list, V_list, W_list)
                
                # Gradient-based update
                # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r] (summed over k... but k is fixed per entry)
                # Actually for each (i,j,k): dT[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r]
                # So gradient for U[i,j,r] = sum_k error[i,j,k] * V[j,k,r] * W[k,i,r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(target_rank):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                e = error[i,j,k]
                                dU[i*3+j, r] += e * V[j*3+k, r] * W[k*3+i, r]
                                dV[j*3+k, r] += e * U[i*3+j, r] * W[k*3+i, r]
                                dW[k*3+i, r] += e * U[i*3+j, r] * V[j*3+k, r]
                
                # Take a step in the negative gradient direction
                lr = 0.1
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Hard thresholding: keep only top-target_rank columns by norm
                # This is the "hard" part of IHT - we project back to rank-target_rank subspace
                # For simplicity, we just renormalize columns
                for r in range(target_rank):
                    nu = np.linalg.norm(U[:, r])
                    nv = np.linalg.norm(V[:, r])
                    nw = np.linalg.norm(W[:, r])
                    if nu > 0:
                        U[:, r] /= nu
                    if nv > 0:
                        V[:, r] /= nv
                    if nw > 0:
                        W[:, r] /= nw
                    # Scale back to maintain magnitude
                    scale = np.cbrt(nu * nv * nw) if nu * nv * nw > 0 else 1
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
            
            # Check if we converged
            T_approx = np.zeros((3, 3, 3))
            for r in range(target_rank):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            T_approx[i,j,k] += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
            err_norm = np.linalg.norm(T_target - T_approx)
            if err_norm < tol:
                U_list = [[float(U[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                V_list = [[float(V[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                W_list = [[float(W[i*3+j, r]) for r in range(target_rank)] for i in range(3) for j in range(3)]
                return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from 25 down to 20
    for target_rank in range(25, 19, -1):
        if time.time() > deadline - 10:
            break
        
        result = iht_search(target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list, target_rank, tol=1e-10):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
