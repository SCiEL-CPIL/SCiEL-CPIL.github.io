import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W give an exact decomposition of the multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # The decomposition says: C[i,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k_c*n+i_c, c]
    # Actually, the tensor for matrix multiplication: T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # The factorization: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check: for each (i,j,k), T[i,j,k] should equal 1 if the term A[i,j]*B[j,k] contributes
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Build the tensor as a 3D array
    # T[i, j, k] = 1 for all valid indices
    # Check using einsum-like computation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # First, get the schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find a lower rank decomposition using gradient descent
    # The tensor we want to factorize: T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from 70 down to 60
    np.random.seed(42)
    
    for target_rank in [70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]:
        if time.time() > deadline - 30:
            break
            
        found = False
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
                
            # Initialize with random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Use Adam-like gradient descent
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            t = 0
            max_iter = 500
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                t += 1
                
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # f = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # We want f = 1
                            f = 0.0
                            for c in range(target_rank):
                                f += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            loss += (f - 1.0) ** 2
                            
                            # Gradients
                            for c in range(target_rank):
                                g = 2.0 * (f - 1.0)
                                dU[i*m+j, c] += g * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += g * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += g * U[i*m+j, c] * V[j*p+k, c]
                
                # Update with Adam
                m_U = beta1 * m_U + (1 - beta1) * dU
                v_U = beta2 * v_U + (1 - beta2) * dU ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * dV
                v_V = beta2 * v_V + (1 - beta2) * dV ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_W = beta2 * v_W + (1 - beta2) * dW ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if it % 50 == 0:
                    if abs(loss - prev_loss) < 1e-10:
                        break
                    prev_loss = loss
                
                # Check convergence
                if loss < 1e-10:
                    break
            
            # Check if we found a valid decomposition
            if _verify(U, V, W, n, m, p, tol=1e-6):
                # Round to nearest half-integer or integer if possible
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_half_int_list(arr):
        """Convert numpy array to list of lists with half-integer values."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                h = round(val * 2) / 2.0
                new_row.append(h)
            result.append(new_row)
        return result
    
    # Verify the best solution once more
    if best_rank < schoolbook_rank:
        U_list = to_half_int_list(best_U)
        V_list = to_half_int_list(best_V)
        W_list = to_half_int_list(best_W)
        
        # Final verification with the rounded values
        U_check = np.array(U_list)
        V_check = np.array(V_list)
        W_check = np.array(W_list)
        if _verify(U_check, V_check, W_check, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    U_list = to_half_int_list(best_U)
    V_list = to_half_int_list(best_V)
    W_list = to_half_int_list(best_W)
    return {"U": U_list, "V": V_list, "W": W_list}
