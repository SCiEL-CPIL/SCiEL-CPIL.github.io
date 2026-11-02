import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    
    schoolbook_rank = n * m * p  # 72
    
    def make_schoolbook():
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication.
        
        Convention: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        For exact verification, we check that for all i,k:
        sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = delta_{i,k} (in the right sense)
        
        Actually, the correct check: the tensor T[i,j,k] = 1 for all i,j,k should equal
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        rank = len(U[0]) if len(U) > 0 else 0
        # Build the decomposition tensor
        # T_decomposed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_decomposition_fast(U_arr, V_arr, W_arr):
        """Fast verification using numpy. U_arr shape (n*m, rank), etc."""
        rank = U_arr.shape[1]
        # T_decomposed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        # Contract over r: result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # First: U_3d[i,j,r] * V_3d[j,k,r] -> sum over r gives (n, m, p) tensor
        # But W_3d has indices (k, i, r), so we need to be careful
        # Let's compute step by step
        # temp[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r]
        temp = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
        # Now multiply by W_3d[k,i,r]... wait, we already summed over r
        # Actually we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a 3-way contraction, not separable that way
        # Let's use: result[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this as: for each r, outer product contribution
        # Or: result = np.einsum('ijr,jkr,kr->ijk', ...) but W is indexed by (k,i,r)
        # W_3d has shape (p, n, rank) with W_3d[k, i, r]
        # So: result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Let's do it in two steps:
        # Step 1: A[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r]  -> shape (n,m,p)
        # Step 2: result[i,j,k] = A[i,j,k] * ... no, that's not right because W depends on r too
        
        # Correct approach: 
        # result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Use einsum: 'ijr,jkr,kr->ijk' won't work directly because of index ordering
        # Let's transpose W_3d to get W_3d_transposed[i,k,r] = W_3d[k,i,r]
        W_3d_t = W_3d.transpose(1, 0, 2)  # shape (n, p, rank), W_3d_t[i,k,r] = W_3d[k,i,r]
        result = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_t)
        return np.allclose(result, 1.0, atol=1e-9)
    
    def to_fraction_list(mat):
        """Convert a numpy array of floats to list of lists of Fractions (if they look rational)."""
        result = []
        for row in mat:
            frow = []
            for val in row:
                # Try to represent as a simple fraction
                if abs(val - round(val)) < 1e-10:
                    frow.append(Fraction(int(round(val))))
                else:
                    # Try common denominators
                    found = False
                    for denom in [2, 3, 4, 5, 6, 8, 10, 12]:
                        num = val * denom
                        if abs(num - round(num)) < 1e-9:
                            frow.append(Fraction(int(round(num)), denom))
                            found = True
                            break
                    if not found:
                        frow.append(Fraction(val).limit_denominator(1000))
            result.append(frow)
        return result
    
    def fraction_to_float_list(frac_mat):
        """Convert list of lists of Fractions to list of lists of floats."""
        return [[float(x) for x in row] for row in frac_mat]
    
    # Strategy: Try randomized search for lower rank decompositions
    # The tensor is T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all 1s)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with schoolbook as best known
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Approach: random initialization + gradient-free optimization
    # Or: structured search based on known patterns
    
    # For M_{3,4,6}, known lower bound is max(12,24,18) = 24
    # Let's try ranks from 24 upward, with limited time per rank
    
    search_ranks = list(range(24, min(40, schoolbook_rank)))
    
    for target_rank in search_ranks:
        if time.time() > deadline - 15:
            break
        
        max_attempts = 50
        attempts = 0
        found = False
        
        while attempts < max_attempts and time.time() < deadline - 15:
            attempts += 1
            
            # Random initialization
            np.random.seed(attempts + target_rank * 1000)
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement
            # Objective: minimize ||sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1||^2
            # Use coordinate descent or simple gradient steps
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Precompute for speed
            U_3d = U_curr.reshape(n, m, target_rank)
            V_3d = V_curr.reshape(m, p, target_rank)
            W_3d = W_curr.reshape(p, n, target_rank)
            W_3d_t = W_3d.transpose(1, 0, 2)  # (n, p, rank)
            
            converged = False
            for step in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Compute current error
                result = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_t)
                error = np.mean((result - 1.0)**2)
                
                if error < 1e-12:
                    converged = True
                    break
                
                # Compute gradients
                # dL/dU_3d[i,j,r] = 2 * (result[i,j,k] - 1) * V_3d[j,k,r] * W_3d_t[i,k,r] summed over k
                # dL/dV_3d[j,k,r] = 2 * (result[i,j,k] - 1) * U_3d[i,j,r] * W_3d_t[i,k,r] summed over i
                # dL/dW_3d_t[i,k,r] = 2 * (result[i,j,k] - 1) * U_3d[i,j,r] * V_3d[j,k,r] summed over j
                
                residual = result - 1.0  # (n, m, p)
                
                # Gradient for U: sum_k residual[i,j,k] * V_3d[j,k,r] * W_3d_t[i,k,r]
                grad_U = np.einsum('ijk,jkr,ikr->ijr', residual, V_3d, W_3d_t)
                # Gradient for V: sum_i residual[i,j,k] * U_3d[i,j,r] * W_3d_t[i,k,r]
                grad_V = np.einsum('ijk,ijr,ikr->jkr', residual, U_3d, W_3d_t)
                # Gradient for W_3d_t: sum_j residual[i,j,k] * U_3d[i,j,r] * V_3d[j,k,r]
                grad_W_t = np.einsum('ijk,ijr,jkr->ikr', residual, U_3d, V_3d)
                
                # Adaptive learning rate
                lr = 0.01 / (1 + step * 0.01)
                
                # Update
                U_curr -= lr * grad_U.reshape(n*m, target_rank)
                V_curr -= lr * grad_V.reshape(m*p, target_rank)
                W_curr -= lr * grad_W_t.reshape(p, n, target_rank).transpose(1, 0, 2).reshape(p*n, target_rank)
                
                # Update 3D views
                U_3d = U_curr.reshape(n, m, target_rank)
                V_3d = V_curr.reshape(m, p, target_rank)
                W_3d = W_curr.reshape(p, n, target_rank)
                W_3d_t = W_3d.transpose(1, 0, 2)
            
            # Check final error
            result_final = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d_t)
            final_error = np.max(np.abs(result_final - 1.0))
            
            if final_error < 1e-8:
                # Found a valid decomposition!
                found = True
                break
        
        if found:
            # Verify thoroughly
            if verify_decomposition_fast(U_curr, V_curr, W_curr):
                # Convert to rational if possible, otherwise use floats
                # The task says integer/half-integer entries, so let's try to rationalize
                U_frac = to_fraction_list(U_curr)
                V_frac = to_fraction_list(V_curr)
                W_frac = to_fraction_list(W_curr)
                
                # Verify with fractions
                U_check = np.array(fraction_to_float_list(U_frac))
                V_check = np.array(fraction_to_float_list(V_frac))
                W_check = np.array(fraction_to_float_list(W_frac))
                
                if verify_decomposition_fast(U_check, V_check, W_check):
                    best_result = {
                        "U": fraction_to_float_list(U_frac),
                        "V": fraction_to_float_list(V_frac),
                        "W": fraction_to_float_list(W_frac)
                    }
                    best_rank = target_rank
                    break  # Found the lowest rank, stop
                else:
                    # Use the float version
                    best_result = {
                        "U": U_curr.tolist(),
                        "V": V_curr.tolist(),
                        "W": W_curr.tolist()
                    }
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
