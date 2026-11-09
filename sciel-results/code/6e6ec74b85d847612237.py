"""Tensor decomposition search for matrix multiplication 3x3 x 3x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal (or best found) decomposition for 3x3 by 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 63
    
    def make_schoolbook():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        r = len(U[0]) if U else 0
        # Reconstruct the tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually, the standard decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor form: T_{i,k}^{j} = A_{i,j} B_{j,k}
        # Decomposition: T = sum_r U^{(r)} \otimes V^{(r)} \otimes W^{(r)}
        # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        # and the indices are: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k) in the A and B tensor, and check the C tensor
        # The multiplication tensor M has entries M[i*m+j, j*p+k] for the A*B product structure
        # Actually let's think more carefully.
        
        # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # As a 3-tensor: T[i,j,k] where T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
        # But the standard decomposition represents:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check: for all i, j, k, the sum should equal 1 if (i,j,k) corresponds to a valid multiplication term
        # The valid terms are all (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
        # Each such term should have value 1 in the tensor.
        
        # Actually, the tensor we need to decompose has:
        # T[i, j, k] = 1 for all valid (i,j,k)
        # And the decomposition is: T = sum_r u_r \otimes v_r \otimes w_r
        # where u_r[i*m+j] = U[i*m+j, r], v_r[j*p+k] = V[j*p+k, r], w_r[k*n+i] = W[k*n+i, r]
        
        # So we need: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for invalid index combinations, it should be 0 (but since we only have valid indices in our arrays, 
        # we just need to check all valid (i,j,k) tuples).
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(r):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best_result = make_schoolbook()
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using random search
    # Strategy: start from a random initialization and try to fit the tensor
    # We'll use a greedy/iterative approach
    
    def try_random_rank(target_rank, num_trials=100, max_iters=500):
        """Try to find a decomposition of the given rank using random initialization and refinement."""
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            rng = np.random.RandomState(trial * 7 + 12345)
            
            # Scale: entries should be reasonable, use integers or half-integers
            # Start with small random values
            scale = 1.0
            U_arr = rng.randint(-3, 4, size=(n*m, target_rank)) * 0.5  # half-integers
            V_arr = rng.randint(-3, 4, size=(m*p, target_rank)) * 0.5
            W_arr = rng.randint(-3, 4, size=(p*n, target_rank)) * 0.5
            
            # Iteratively refine using gradient-like updates
            # Objective: minimize sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # plus penalty for non-integer/half-integer values
            
            best_loss = float('inf')
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor values
                tensor = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U_arr[:, r].reshape(n, m)
                    v = V_arr[:, r].reshape(m, p)
                    w = W_arr[:, r].reshape(p, n)
                    # Outer product: u[i,j] * v[j,k] * w[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                # Compute loss
                loss = np.sum((tensor - 1.0) ** 2)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_arr * 2) / 2
                V_snap = np.round(V_arr * 2) / 2
                W_snap = np.round(W_arr * 2) / 2
                
                tensor_snap = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U_snap[:, r].reshape(n, m)
                    v = V_snap[:, r].reshape(m, p)
                    w = W_snap[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor_snap[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                loss_snap = np.sum((tensor_snap - 1.0) ** 2)
                
                if loss_snap < best_loss:
                    best_loss = loss_snap
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                
                if loss_snap < 1e-6:
                    break
                
                # Gradient computation and update
                # For U[i*m+j, r]:
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            idx = i * m + j
                            grad = 0.0
                            for k in range(p):
                                # d/dU[idx,r] of (sum_r' U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - 1)^2
                                # = 2 * (tensor[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                                val = tensor[i, j, k] - 1.0
                                grad += 2 * val * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                            U_arr[idx, r] -= 0.01 * grad  # small step
                    
                    for j in range(m):
                        for k in range(p):
                            idx = j * p + k
                            grad = 0.0
                            for i in range(n):
                                val = tensor[i, j, k] - 1.0
                                grad += 2 * val * U_arr[i*m+j, r] * W_arr[k*n+i, r]
                            V_arr[idx, r] -= 0.01 * grad
                    
                    for k in range(p):
                        for i in range(n):
                            idx = k * n + i
                            grad = 0.0
                            for j in range(m):
                                val = tensor[i, j, k] - 1.0
                                grad += 2 * val * U_arr[i*m+j, r] * V_arr[j*p+k, r]
                            W_arr[idx, r] -= 0.01 * grad
                
                # Occasionally restart if stuck
                if it % 100 == 99:
                    current_loss = np.sum((tensor - 1.0)**2)
                    if current_loss > best_loss * 1.1:
                        U_arr = best_U + rng.randn(n*m, target_rank) * 0.1
                        V_arr = best_V + rng.randn(m*p, target_rank) * 0.1
                        W_arr = best_W + rng.randn(p*n, target_rank) * 0.1
            
            # Check if we found a valid solution
            if best_loss < 1e-9:
                # Convert to list format with exact half-integer values
                U_list = [[0] * target_rank for _ in range(n*m)]
                V_list = [[0] * target_rank for _ in range(m*p)]
                W_list = [[0] * target_rank for _ in range(p*n)]
                for i in range(n*m):
                    for r in range(target_rank):
                        U_list[i][r] = best_U[i, r]
                for i in range(m*p):
                    for r in range(target_rank):
                        V_list[i][r] = best_V[i, r]
                for i in range(p*n):
                    for r in range(target_rank):
                        W_list[i][r] = best_W[i, r]
                
                # Final verification
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks lower than schoolbook
    # Known lower bounds: R(3,3,3) = 23, so R(3,3,7) >= 23
    # Try a few promising ranks
    target_ranks = [59, 56, 54, 50, 46, 42, 38, 34, 30, 26, 23]
    
    for tr in target_ranks:
        if time.time() > deadline:
            break
        if tr >= best_rank:
            continue
        result = try_random_rank(tr, num_trials=20, max_iters=300)
        if result is not None:
            best_result = result
            best_rank = tr
            break  # Found a valid lower rank, use it
    
    # If nothing better found, return schoolbook
    if best_rank == rank_school:
        return make_schoolbook()
    else:
        return best_result

# EVOLVE-BLOCK-END
