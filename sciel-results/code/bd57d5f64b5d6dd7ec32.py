import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # The matrix multiplication tensor T[i,j,k] = A[i,j]*B[j,k]
    # We want: T = sum_{t=1}^{rank} u_t[i,j] * v_t[j,k] * w_t[k,i]
    # where u_t is a vector of length n*m, v_t of length m*p, w_t of length p*n
    # W is indexed by k*n + i
    
    # Trivial rank is n*m*p = 40
    # Current best known: 36
    
    # Let's try to find a lower rank decomposition using optimization
    # We'll use a randomized approach with local optimization
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Precompute the tensor
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: for each (i,k), the sum over j of A[i,j]*B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 (it's the "multiplication" tensor)
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] such that C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    # Here T[i,j,k] = 1 for all i,j,k (since we just sum the products)
    # Wait, no. The tensor is defined by the structure of the multiplication.
    
    # Let me re-derive. We have:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # A decomposition of rank r means:
    # C[i,k] = sum_{t=1}^{r} (sum_j U[t, i*m+j]) * (sum_k V[t, j*p+k]) * W[t, k*n+i]
    # 
    # Wait, that's not quite right either. Let me be more careful.
    #
    # U has shape (n*m, r): U[i*m+j, t]
    # V has shape (m*p, r): V[j*p+k, t]  
    # W has shape (p*n, r): W[k*n+i, t]
    #
    # The decomposition says:
    # For all i, k: C[i,k] = sum_{t=1}^{r} W[t, k*n+i] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    #
    # No wait, that's not right. The standard tensor decomposition for matrix multiplication is:
    # 
    # We want: sum_j A[i,j] B[j,k] = sum_t (sum_j U[i*m+j, t] A[i,j]) (sum_j V[j*p+k, t] B[j,k]) W[t, k*n+i]
    #
    # Hmm, let me look at this more carefully.
    
    # The correct formulation:
    # A is n×m, B is m×p, C = A*B is n×p
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition gives:
    # C[i,k] = sum_{t=1}^{r} M_t * W[t, k*n+i]
    # where M_t = (sum_j U[t, i*m+j] * A[i,j]) * (sum_j V[t, j*p+k] * B[j,k])
    #
    # Wait, that doesn't look right dimensionally. Let me think again.
    
    # Actually, the standard way:
    # U: (n*m) x r matrix, so U[row, t] where row = i*m + j
    # V: (m*p) x r matrix, so V[row, t] where row = j*p + k  
    # W: (p*n) x r matrix, so W[row, t] where row = k*n + i
    #
    # The decomposition: for all i, j, k:
    # delta_{i,j,k} (the multiplication tensor) = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # No, that's not right either. The multiplication tensor T is defined by:
    # T[i, j, k] = 1 (for the "1" in A[i,j]*B[j,k] contributing to C[i,k])
    #
    # Actually, I think the correct statement is:
    # The matrix multiplication map is a bilinear map: (A, B) -> C
    # where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{t=1}^{r} W[k*n+i, t] * (U[i*m+?, t] ... )
    #
    # Let me look at this from the tensor perspective.
    # 
    # The multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # is defined by: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # Wait, that's not right. Let me be very precise.
    #
    # We have: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This can be written as: C = T(A, B) where T is a trilinear form... no.
    #
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-1 decomposition would be: C[i,k] = f(A) * g(B) * h_{i,k}
    # where f(A) = sum_{i,j} U[i*m+j] A[i,j], g(B) = sum_{j,k} V[j*p+k] B[j,k], h_{i,k} = W[k*n+i]
    #
    # So: C[i,k] = (sum_{i',j} U[i'*m+j] A[i',j]) * (sum_{j',k'} V[j'*p+k'] B[j',k']) * W[k*n+i]
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_{i',j} U[i'*m+j] A[i',j] * sum_{j',k'} V[j'*p+k'] B[j',k'] * W[k*n+i] = sum_j A[i,j] B[j,k]
    #
    # This must hold for all A and B. Expanding:
    # sum_{i',j,j',k'} U[i'*m+j] V[j'*p+k'] W[k*n+i] A[i',j] B[j',k'] = sum_j A[i,j] B[j,k]
    #
    # The RHS is: sum_j A[i,j] B[j,k] = sum_{i',j,j',k'} delta_{i',i} delta_{j,j'} delta_{k',k} A[i',j] B[j',k']
    #
    # So we need: sum_t U_t[i*m+j] V_t[j*p+k] W_t[k*n+i] = delta_{i,j,k}... 
    #
    # Hmm, I'm getting confused. Let me just use the standard formulation.
    
    # Standard: The matrix multiplication tensor for (n,m,p) is:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T is a 3-tensor of size (n*m) x (m*p) x (p*n)? No...
    
    # OK let me just use the direct approach. We want to find U, V, W such that:
    # For all i in [0,n), k in [0,p):
    #   sum_j A[i,j] B[j,k] = sum_t W[t, k*n+i] * (sum_j U[t, i*m+j] * A[i,j]) * (sum_j V[t, j*p+k] * B[j,k])
    #
    # Wait, that gives a product of two sums, which is a double sum, not matching.
    
    # I think the correct formulation is:
    # C[i,k] = sum_t (sum_j U[t, i*m+j] * A[i,j]) * (sum_j V[t, j*p+k] * B[j,k]) * W[t, k*n+i]
    #
    # But (sum_j U[t, i*m+j] * A[i,j]) is a scalar (depends on i and t),
    # (sum_j V[t, j*p+k] * B[j,k]) is a scalar (depends on k and t),
    # and W[t, k*n+i] is a scalar.
    #
    # So C[i,k] = sum_t scalar1(i,t) * scalar2(k,t) * W[t,k*n+i]
    #
    # But we need C[i,k] = sum_j A[i,j] B[j,k], which is bilinear in A and B.
    # The expression sum_t f(i,t,A) * g(k,t,B) * W[t,k*n+i] is also bilinear in A and B.
    # 
    # For this to work for all A, B, we need the coefficients to match.
    
    # Let me expand:
    # sum_t (sum_{j1} U[t, i*m+j1] A[i,j1]) * (sum_{j2} V[t, j2*p+k] B[j2,k]) * W[t, k*n+i]
    # = sum_t sum_{j1,j2} U[t, i*m+j1] V[t, j2*p+k] W[t, k*n+i] A[i,j1] B[j2,k]
    #
    # We need this to equal sum_j A[i,j] B[j,k] for all i,k,A,B.
    #
    # So: sum_t sum_{j1,j2} U[t, i*m+j1] V[t, j2*p+k] W[t, k*n+i] delta_{j1,j2} ... 
    #
    # No, we need: for each (i, j1, j2, k), the coefficient of A[i,j1] B[j2,k] must be:
    # 1 if j1 = j2 (and it contributes to C[i,k]), 0 otherwise.
    #
    # Wait, A[i,j1] B[j2,k] only contributes to C[i,k] if j1 = j2. And it contributes with coefficient 1.
    #
    # So we need: for all i, k, j1, j2:
    # sum_t U[t, i*m+j1] * V[t, j2*p+k] * W[t, k*n+i] = delta_{j1,j2}
    #
    # That's the condition! Let me verify with the trivial decomposition.
    # Trivial: r = n*m*p. For each (i,j,k), have one term with:
    # U[t, i*m+j] = 1, V[t, j*p+k] = 1, W[t, k*n+i] = 1
    # Then sum_t ... = 1 only when the specific (i,j,k) matches, giving delta_{j1,j2} when j1=j2=j and i,k match.
    # Yes, that works.
    
    # So the condition is:
    # For all i in [0,n), k in [0,p), j1 in [0,m), j2 in [0,m):
    #   sum_{t=1}^{r} U[t, i*m+j1] * V[t, j2*p+k] * W[t, k*n+i] = delta_{j1,j2}
    
    # This is a system of equations. Let's set it up.
    
    # Number of equations: n * p * m * m = 2 * 5 * 4 * 4 = 160
    # Unknowns for rank r: r * (n*m + m*p + p*n) = r * (8 + 20 + 10) = 38r
    # For r=36: 1368 unknowns, 160 equations. Very underdetermined.
    
    # Let's try to find a decomposition of rank r using optimization.
    # We'll minimize the sum of squared residuals.
    
    def check_decomposition(U, V, W, r):
        """Check if the decomposition is exact."""
        # U: (r, n*m), V: (r, m*p), W: (r, p*n)
        # Condition: for all i,k,j1,j2: sum_t U[t,i*m+j1]*V[t,j2*p+k]*W[t,k*n+i] = delta_{j1,j2}
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for t in range(r):
                            s += U[t, i*m+j1] * V[t, j2*p+k] * W[t, k*n+i]
                        expected = 1.0 if j1 == j2 else 0.0
                        err = abs(s - expected)
                        if err > max_err:
                            max_err = err
        return max_err
    
    def check_decomposition_fast(U, V, W, r):
        """Faster check using numpy."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                # For fixed i,k, we need: for all j1,j2: sum_t U[t,i*m+j1]*V[t,j2*p+k]*W[t,k*n+i] = delta_{j1,j2}
                # Let a_j1 = U[:, i*m+j1] (vector of length r)
                # Let b_j2 = V[:, j2*p+k] (vector of length r)  
                # Let w = W[:, k*n+i] (vector of length r)
                # We need: a_j1^T * (w * b_j2) = delta_{j1,j2}
                # i.e., (a_j1 * w)^T * b_j2 = delta_{j1,j2}
                # i.e., the matrix with rows (a_j1 * w)^T and columns b_j2 should be I_m
                A_i = np.array([U[:, i*m+j] * W[:, k*n+i] for j in range(m)])  # (m, r)
                B_k = np.array([V[:, j*p+k] for j in range(m)])  # (m, r)
                M = A_i @ B_k.T  # (m, m)
                err = np.max(np.abs(M - np.eye(m)))
                if err > max_err:
                    max_err = err
        return max_err
    
    # Try ranks from low to high
    best_result = None
    best_rank = None
    
    # Start with known good rank 36, try to find lower
    for target_rank in [30, 32, 34, 35, 36]:
        if time.time() > deadline:
            break
        
        found = False
        num_trials = 50
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            r = target_rank
            # Random initialization
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(r, n*m)
            V = np.random.randn(r, m*p)
            W = np.random.randn(r, p*n)
            
            # Optimize using gradient descent
            # Loss = sum of squared residuals
            def loss_and_grad(U, V, W):
                loss = 0.0
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        A_i = np.array([U[:, i*m+j] * W[:, k*n+i] for j in range(m)])  # (m, r)
                        B_k = np.array([V[:, j*p+k] for j in range(m)])  # (m, r)
                        M = A_i @ B_k.T  # (m, m)
                        target = np.eye(m)
                        err = M - target  # (m, m)
                        loss += np.sum(err**2)
                        
                        # Gradients
                        # M[j1,j2] = sum_t U[t,i*m+j1]*W[t,k*n+i]*V[t,j2*p+k]
                        # dM[j1,j2]/dU[t,i*m+j1'] = delta_{j1,j1'} * W[t,k*n+i] * V[t,j2*p+k]
                        # dM[j1,j2]/dV[t,j2'*p+k] = delta_{j2,j2'} * U[t,i*m+j1] * W[t,k*n+i]
                        # dM[j1,j2]/dW[t,k*n+i'] = delta_{k*n+i, k*n+i'} * U[t,i*m+j1] * V[t,j2*p+k]
                        
                        for j1 in range(m):
                            for j2 in range(m):
                                e = err[j1, j2]
                                for t in range(r):
                                    # dU
                                    gU[t, i*m+j1] += 2*e * W[t, k*n+i] * V[t, j2*p+k]
                                    # dV
                                    gV[t, j2*p+k] += 2*e * U[t, i*m+j1] * W[t, k*n+i]
                                    # dW
                                    gW[t, k*n+i] += 2*e * U[t, i*m+j1] * V[t, j2*p+k]
                return loss, gU, gV, gW
            
            # Simple gradient descent with adaptive learning rate
            lr = 0.01
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                loss, gU, gV, gW = loss_and_grad(U, V, W)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Adaptive learning rate
                if loss < best_loss * 0.99:
                    lr = min(lr * 1.1, 0.1)
                else:
                    lr = max(lr * 0.9, 0.001)
            
            # Check the best found
            if best_loss < 1e-8:
                # Try to round to rationals
                # Scale to make entries nice
                max_abs = max(np.max(np.abs(best_U)), np.max(np.abs(best_V)), np.max(np.abs(best_W)))
                if max_abs > 0:
                    scale = max_abs
                    U_r = best_U / scale
                    V_r = best_V / scale
                    W_r = best_W * scale  # Keep the product the same
                
                # Try rounding to half-integers
                for mult in [1, 2, 4, 8]:
                    U_round = np.round(best_U * mult) / mult
                    V_round = np.round(best_V * mult) / mult
                    W_round = np.round(best_W * mult) / mult
                    err = check_decomposition_fast(U_round, V_round, W_round, r)
                    if err < 1e-9:
                        # Found an exact decomposition!
                        # Convert to lists
                        U_list = [[float(U_round[t, i*m+j]) for j in range(m)] for i in range(n)]
                        # Reshape to (n*m, r)
                        U_out = [[0.0]*r for _ in range(n*m)]
                        V_out = [[0.0]*r for _ in range(m*p)]
                        W_out = [[0.0]*r for _ in range(p*n)]
                        for t in range(r):
                            for i in range(n):
                                for j in range(m):
                                    U_out[i*m+j][t] = U_round[t, i*m+j]
                            for j in range(m):
                                for k in range(p):
                                    V_out[j*p+k][t] = V_round[t, j*p+k]
                            for k in range(p):
                                for i in range(n):
                                    W_out[k*n+i][t] = W_round[t, k*n+i]
                        
                        if best_rank is None or r < best_rank:
                            best_rank = r
                            best_result = {"U": U_out, "V": V_out, "W": W_out}
                        found = True
                        break
                if found:
                    break
        
        if found:
            break
    
    # If no lower rank found, use the known rank-36 decomposition
    if best_result is None:
        # Use the original rank-36 decomposition
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j):
            return i * m + j
        def v_idx(j, k):
            return j * p + k
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L B_15
        U[u_idx(0,0)][col] = 1
        V[v_idx(0,4)][col] = 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(0,1)][col] = 1
        V[v_idx(1,4)][col] = 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(1,0)][col] = 1
        V[v_idx(0,4)][col] = 1
        W[w_idx(4,1)][col] += 1
        col += 1
        U[u_idx(1,1)][col] = 1
        V[v_idx(1,4)][col] = 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        # Block 6: A_R B_25
        U[u_idx(0,2)][col] = 1
        V[v_idx(2,4)][col] = 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(0,3)][col] = 1
        V[v_idx(3,4)][col] = 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(1,2)][col] = 1
        V[v_idx(2,4)][col] = 1
        W[w_idx(4,1)][col] += 1
        col += 1
        U[u_idx(1,3)][col] = 1
        V[v_idx(3,4)][col] = 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        return best_result

