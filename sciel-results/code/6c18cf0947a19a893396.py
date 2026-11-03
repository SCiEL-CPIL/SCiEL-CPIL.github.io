import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if indices match for matmul, else 0
    # Actually, the tensor is T[(i,j), (j',k'), (k'',i'')] = delta(j,j') delta(k',k'') delta(i,i'')
    # We want sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] = T
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # Known rank for 3x3x4 is 27
    target_rank = 27
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    best_result = None
    best_rank = rank_school
    
    # Generate schoolbook solution as fallback
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_school[i*m+j, r] = 1
                V_school[j*p+k, r] = 1
                W_school[k*n+i, r] = 1
                
    # Function to verify a solution
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] == delta(j,j') delta(k',k'') delta(i,i'')
        # Flatten indices
        # i in [0,n), j in [0,m), k in [0,p)
        # U index: i*m+j
        # V index: j*p+k
        # W index: k*n+i
        # The tensor entry T[i,j,k] is 1 if we are forming C[i,k] from A[i,j] and B[j,k].
        # The condition is:
        # For all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 ?
        # No, the tensor is defined such that:
        # sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] = 1 if j=j', k'=k'', i=i'' else 0.
        # Let's check the indices.
        # If we set A[i,j] and B[j',k'], the product term is U[i,j,r] V[j',k',r].
        # The output is C[k'',i'']? No, C[i'',k''].
        # The W index is k''*n + i''.
        # So we need:
        # sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r] = 1 if j=j', k'=k'', i=i'' else 0.
        
        # Let's construct the tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term A[i,j] * B[j,k] contributes to C[i,k]
                    # So U index is i*m+j
                    # V index is j*p+k
                    # W index is k*n+i
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute the reconstructed tensor
        # Rec[i,j, v_idx, w_idx] = sum_r U[i,j,r] V[v_idx,r] W[w_idx,r]
        # This is a 4-tensor. We need to check if it matches T.
        # Instead of full 4-tensor, we can check the contraction.
        # But T is sparse.
        # Let's just compute the error on the non-zero entries of T.
        # There are n*m*p = 36 non-zero entries.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # Sum over r
                    s = np.dot(U[u_idx, :], V[v_idx, :]) * 0 # Wrong, need element-wise product sum
                    # s = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # This is not a simple dot product.
                    # It is the inner product of (U[u_idx,:] * V[v_idx,:]) with W[w_idx,:]
                    s = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                    error += (s - 1.0)**2
        
        # Also need to check that the "off-diagonal" entries are 0?
        # The tensor T has 1s only at (i*m+j, j*p+k, k*n+i).
        # All other entries must be 0.
        # Checking all entries is (n*m)*(m*p)*(p*n) = 9*12*12 = 1296 entries.
        # This is fast.
        
        # Reconstruct full tensor?
        # Rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is expensive if done naively.
        # But R is small (27).
        # We can do:
        # Rec = np.einsum('ar,br,cr->abc', U, V, W)
        # Then check np.allclose(Rec, T)
        
        # Let's do this.
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(Rec, T, atol=1e-9):
            return False
        return True

    # Try to find a rank 27 solution
    # Use a simple iterative method or random search with local improvement?
    # Given the time limit, a robust numerical solver for exact integer solutions is hard.
    # I will try a few random initializations and use gradient descent to minimize error.
    
    def solve_rank(R):
        if R >= best_rank:
            return None
        if time.time() > deadline:
            return None
            
        # Random initialization
        # Use small integers
        U = np.random.randint(-2, 3, (n*m, R))
        V = np.random.randint(-2, 3, (m*p, R))
        W = np.random.randint(-2, 3, (p*n, R))
        
        # Gradient descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                return None
                
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((Rec - T)**2)
            if error < 1e-9:
                # Check if integers/half-integers
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
                return None
            
            # Compute gradients
            # dE/dU[a, r] = 2 * sum_{b,c} (Rec[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dE/dV[b, r] = 2 * sum_{a,c} (Rec[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dE/dW[c, r] = 2 * sum_{a,b} (Rec[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            diff = Rec - T
            
            # dU
            # dU[a, r] = 2 * sum_b diff[a, b, :] * V[b, r] * W[:, r]? No.
            # dU[a, r] = 2 * sum_{b,c} diff[a, b, c] * V[b, r] * W[c, r]
            # This is: 2 * (diff @ (V * W)?)
            # Let's use einsum
            dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return None

    # Try ranks from 27 up to 35
    for R in range(27, min(36, best_rank)):
        if time.time() > deadline:
            break
        res = solve_rank(R)
        if res:
            U, V, W = res
            best_rank = R
            best_result = (U, V, W)
            break # Found a lower rank, stop
            
    if best_result:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
