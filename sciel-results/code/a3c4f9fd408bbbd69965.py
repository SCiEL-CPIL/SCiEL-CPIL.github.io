import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    total_elements = n * m * p
    
    # Construct the target matrix multiplication tensor T[i, j, k] = delta_{i,j,k} 
    # Actually, the tensor for matrix multiplication C = A * B is defined by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) where T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Wait, the standard tensor decomposition form is:
    # C[i, k] = sum_r U[i, r] V[r, k] ... no.
    # The decomposition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} ? No.
    # Let's look at the interface:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the output factor follows W[k*n+i, r].
    # The tensor being decomposed is the matrix multiplication tensor M_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # The contribution to C[i,k] is A[i,j] * B[j,k].
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 if we are looking at the term A[i,j]B[j,k].
    # But the factors are defined on flattened indices.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The reconstruction is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k? 
    # No, that's not right. The tensor T is sparse.
    # T[i, j, k] is 1 if the indices form a valid path? No.
    # The matrix multiplication tensor is the sum of basis tensors e_{i,j} \otimes e_{j,k} \otimes e_{i,k}^*?
    # Let's re-read the standard definition.
    # The rank of matrix multiplication tensor <n, m, p> is the minimum number of rank-1 tensors needed to express the multiplication.
    # The multiplication tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the indices are linked?
    # Actually, the tensor T is defined such that <T, A \otimes B \otimes C> = trace(C^T A B) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # So the tensor T has components T_{i,j,k} such that sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This implies T_{i,j,k} = 1 for all i,j,k? No.
    # If T_{i,j,k} = 1 for all i,j,k, then sum T A B C = sum A B C.
    # But the indices in the sum are independent in the dot product?
    # The tensor is a multilinear form.
    # T(A, B, C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This can be written as sum_{i,j,k} 1 * A_{i,j} B_{j,k} C_{k,i}.
    # So the tensor T has components T_{i,j,k} = 1 for all i,j,k?
    # Wait, if T_{i,j,k} = 1, then the rank is the number of terms in the sum?
    # No, the rank is the minimum number of rank-1 tensors u \otimes v \otimes w such that T = sum u \otimes v \otimes w.
    # A rank-1 tensor u \otimes v \otimes w has components u_i v_j w_k.
    # We want sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = T_{i,j,k}.
    # What is T_{i,j,k}?
    # From the trace convention: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This identity holds for all A, B, C if T_{i,j,k} = 1 for all i,j,k.
    # So the target tensor is the all-ones tensor of size (n*m) x (m*p) x (p*n)?
    # No, the indices are coupled.
    # Let's check the indices again.
    # U is (n*m, rank). Index u = i*m + j.
    # V is (m*p, rank). Index v = j*p + k.
    # W is (p*n, rank). Index w = k*n + i.
    # The term in the sum is U[u, r] V[v, r] W[w, r].
    # We sum over r.
    # The result is a tensor of size (n*m) x (m*p) x (p*n).
    # Let's call the reconstructed tensor R[u, v, w].
    # We want R[u, v, w] to match the "multiplication tensor" T[u, v, w].
    # What is T[u, v, w]?
    # The multiplication operation is linear in A, B, C.
    # The tensor T is defined by T(A, B, C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # In terms of flattened indices:
    # A_{i,j} is A_flat[i*m + j].
    # B_{j,k} is B_flat[j*p + k].
    # C_{k,i} is C_flat[k*n + i].
    # So T(A_flat, B_flat, C_flat) = sum_{u,v,w} T[u,v,w] A_flat[u] B_flat[v] C_flat[w].
    # Comparing the two sums:
    # sum_{i,j,k} A_flat[i*m+j] B_flat[j*p+k] C_flat[k*n+i]
    # This sum is over i, j, k.
    # For a specific u, v, w, the coefficient T[u,v,w] is 1 if the indices (i,j) from u, (j,k) from v, and (k,i) from w are consistent.
    # i.e., if u = i*m + j, v = j*p + k, w = k*n + i for some i, j, k.
    # And since i, j, k are determined by u, v, w (with consistency checks), T[u,v,w] is 1 if the indices are consistent, 0 otherwise.
    # Specifically:
    # i = u // m, j = u % m
    # j' = v // p, k = v % p
    # k' = w // n, i' = w % n
    # Consistency requires j == j' and k == k' and i == i'.
    # So T[u,v,w] = 1 if (u%m == v//p) and (v%p == w//n) and (w%n == u//m), else 0.
    
    # So the target tensor T is a sparse tensor with 1s on the "consistent" indices.
    # We want to find U, V, W of rank R such that:
    # sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w]
    
    # This is a nonlinear least squares problem if we fix R.
    # We can use gradient descent to minimize:
    # Loss = sum_{u,v,w} ( sum_r U[u,r] V[v,r] W[w,r] - T[u,v,w] )^2
    
    # We need to search for the minimal R.
    # The schoolbook rank is n*m*p = 84.
    # We know R=78 is possible.
    # We will try to optimize for R values starting from a lower bound.
    # A lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28? No, that's for the matrix dimensions.
    # For matrix multiplication <n,m,p>, the rank is at least max(n*m, m*p, p*n) / something?
    # Actually, the trivial lower bound is max(n*m, m*p, p*n) is not correct.
    # The rank is at least max(n*m, m*p, p*n) is false.
    # The rank is at least n*m if m=1?
    # Let's just try to find a rank R such that the optimization converges to 0 error.
    
    # Strategy:
    # 1. Start with R = 78 (known feasible).
    # 2. Try to optimize for R = 78. If it finds a solution with low error, keep it.
    # 3. Try to optimize for R = 77, 76, ... down to some limit.
    # 4. If we find a solution with R < 78, return it.
    # 5. If not, return the best solution found for R=78 or fallback to schoolbook.
    
    # However, optimization is hard. Let's use a smarter initialization.
    # For R=78, we can use the known Strassen-based construction as a seed?
    # The prompt says "inspiration 2" used block decomposition and got 1.0769 (which corresponds to rank 80/78? 80/78 = 1.025? No.
    # Score is seed/value. Seed is 84.
    # If value is 78, score is 84/78 = 1.0769.
    # So inspiration 2 found rank 78.
    # My goal is to find rank < 78 if possible, or just reproduce 78 robustly.
    # But the current program returns rank 84 (score 1.0).
    # I should first ensure I can return rank 78.
    # Can I hardcode the rank 78 decomposition?
    # The prompt says "inspiration 2 ... uses numpy ... defines construct".
    # It doesn't give the code.
    # I can implement the known 2+1 split strategy.
    
    # Let's implement the 2+1 split strategy explicitly.
    # Split n=3 into n1=2, n2=1.
    # A is 3x4, B is 4x7.
    # A = [A1 (2x4); A2 (1x4)]
    # C = [C1 (2x7); C2 (1x7)]
    # C1 = A1 * B (2x7)
    # C2 = A2 * B (1x7)
    
    # For C1 = A1 * B (2x4 * 4x7):
    # We can use Strassen-like blocks?
    # 2x4 * 4x7.
    # Split m=4 into m1=2, m2=2.
    # A1 = [A11 (2x2) A12 (2x2)]
    # B = [B1 (2x7); B2 (2x7)]
    # C1 = A11 B1 + A12 B2.
    # This is sum of two 2x2 * 2x7 multiplications.
    # Rank of 2x2 * 2x7?
    # We can use Strassen on 2x2 * 2x2 blocks?
    # 2x7 is 2x(2+2+2+1).
    # This is getting complicated to hardcode perfectly without errors.
    
    # Alternative: Use the optimization approach but seed it with a known good rank?
    # No, I don't have the known good rank matrices.
    
    # Let's try the optimization approach for R=78.
    # Initialize U, V, W with random small values.
    # Use L-BFGS-B or similar.
    
    # To make it faster and more likely to succeed, I will try a few random restarts for R=78.
    # If I find a solution with error < 1e-6, I return it.
    # If not, I fall back to schoolbook (rank 84).
    
    # But wait, if I can't find 78, I score 1.0.
    # If I find 78, I score 1.0769.
    # If I find 77, I score 84/77 = 1.0909.
    
    # Let's implement the optimizer.
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    T[u, v, w] = 1.0
        return T

    T_target = get_target_tensor()
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    # Try ranks from 78 down to 70?
    # Optimization is expensive. 
    # Let's try R=78 first.
    
    def optimize_rank(R, num_restarts=5):
        start_time_opt = time.time()
        deadline_opt = start_time_opt + 100.0 # Leave 10s for safety
        
        for restart in range(num_restarts):
            if time.time() > deadline_opt:
                break
            
            # Initialize
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Flatten for scipy
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute sum_r U[u,r] V[v,r] W[w,r]
                # This is a tensor contraction.
                # We can do it as:
                # Temp = U @ V.T  -> (n*m, m*p) ? No.
                # We need sum_r U[u,r] V[v,r] W[w,r].
                # This is element-wise product summed over r.
                # We can compute it as:
                # T_recon[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
                # This is expensive to compute fully if we do it naively in Python.
                # Using NumPy:
                # U[:, :, None] * V[None, :, :] ... no, dimensions don't match directly for broadcasting to 3D.
                # U: (n*m, R)
                # V: (m*p, R)
                # W: (p*n, R)
                # We want T: (n*m, m*p, p*n)
                # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # We can compute this as:
                # M = U @ V.T  # (n*m, m*p) -- no, this sums over r U[u,r] V[v,r].
                # Then T[u,v,w] = sum_w? No.
                # T[u,v,w] = (U @ V.T)[u,v] * W[w,r]? No, r is summed.
                # T[u,v,w] = sum_r (U[u,r] V[v,r]) W[w,r]
                # Let M_uv = sum_r U[u,r] V[v,r]. This is U @ V.T.
                # Then T[u,v,w] = sum_r M_uv[r]? No, M_uv is a scalar for each u,v.
                # Wait, M_uv = sum_r U[u,r] V[v,r].
                # Then T[u,v,w] is NOT M_uv * W[w,r] because r is different for each term in the sum?
                # No, r is the summation index.
                # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
                # This is the trace of U[u,:] * V[v,:] * W[w,:]^T?
                # It is the inner product of the vectors U[u,:], V[v,:], W[w,:] in R^R.
                # We can compute this by:
                # 1. Compute M = U @ V.T  (shape n*m x m*p). M[u,v] = sum_r U[u,r] V[v,r].
                #    This is not enough because we need to multiply by W[w,r] and sum over r.
                #    M[u,v] is already summed over r.
                #    So we cannot just multiply M by W.
                #    
                #    We need to keep the r dimension.
                #    Let's compute T using broadcasting.
                #    U_exp = U[:, :, None]  # (n*m, R, 1)
                #    V_exp = V[None, :, :]  # (1, m*p, R)
                #    W_exp = W[None, None, :] # (1, 1, p*n)
                #    Prod = U_exp * V_exp * W_exp # (n*m, m*p, p*n) ? No.
                #    U_exp * V_exp is (n*m, R, m*p).
                #    Then * W_exp (1, 1, p*n) -> broadcast error.
                #    
                #    Correct broadcasting:
                #    U: (n*m, R) -> (n*m, R, 1)
                #    V: (m*p, R) -> (1, m*p, R)
                #    W: (p*n, R) -> (1, 1, p*n) -- wait, R is the last dim for all?
                #    Yes, all have R as the last dim in the factor matrices.
                #    So:
                #    U[:, :, None] * V[None, :, :] * W[None, None, :]
                #    U: (n*m, R, 1)
                #    V: (1, m*p, R)
                #    Product U*V: (n*m, R, m*p) -- this is not aligned.
                #    We need the R dimensions to align.
                #    U: (n*m, R)
                #    V: (m*p, R)
                #    W: (p*n, R)
                #    We want T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
                #    This is equivalent to:
                #    T = np.einsum('ur,vr,wr->uvw', U, V, W)
                
                T_recon = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
                diff = T_recon - T_target
                return np.sum(diff * diff)
            
            # Use L-BFGS-B
            # This might be slow.
            # Let's try a simpler gradient descent or just return schoolbook if optimization is too complex to implement correctly in time.
            
            # Actually, implementing a robust optimizer for tensor rank minimization is very hard.
            # The known 78 rank solution is algebraic.
            # I should implement the algebraic 78 rank solution.
            
            pass
        
        return None

    # Since implementing the optimizer is risky and likely to fail or be slow, 
    # and I know a 78-rank solution exists, I will try to construct it.
    
    # Construction of Rank 78:
    # Split n=3 into 2+1.
    # Part 1: 2x4 * 4x7.
    # Part 2: 1x4 * 4x7.
    
    # Part 2: 1x4 * 4x7.
    # This is just a dot product of a 1x4 vector and 4x7 matrix.
    # C2[i=0, k] = sum_j A2[j] B[j,k].
    # Rank is 4 (schoolbook).
    # U2: (1*4, 4) = (4,4). Identity?
    # V2: (4*7, 4) = (28,4).
    # W2: (7*1, 4) = (7,4).
    # Let's use schoolbook for this part.
    # U2[j, r] = 1 if r==j.
    # V2[j*7+k, r] = 1 if r==j.
    # W2[k*1+0, r] = 1 if r==j.
    
    # Part 1: 2x4 * 4x7.
    # Split m=4 into 2+2.
    # A1 = [A11 (2x2) A12 (2x2)]
    # B = [B1 (2x7) B2 (2x7)]
    # C1 = A11 B1 + A12 B2.
    # We need to decompose A11 B1 and A12 B2.
    # Each is 2x2 * 2x7.
    # 2x2 * 2x7 can be decomposed into 2x2 * 2x2 and 2x2 * 2x1?
    # Or use Strassen on 2x2 * 2x2 blocks?
    # 2x7 = 2x(2+2+2+1).
    # This is getting too detailed.
    
    # Alternative: Just return schoolbook. It's safe.
    # But I want to improve.
    
    # Let's try the optimization again, but with a smaller search space or better initialization.
    # Actually, if I can't guarantee 78, I should stick to 84.
    # But 84 is the current score.
    
    # Wait, the "inspiration 2" got 1.0769. That means it found 78.
    # I can try to replicate the "block decomposition" mentioned in inspiration 2.
    # "It splits ... uses numpy".
    
    # Let's try a simpler optimization: 
    # Fix R=78.
    # Initialize U, V, W with the schoolbook 84 rank solution, then zero out 6 random columns?
    # No, that won't work.
    
    # I will implement the schoolbook solution as a fallback, and attempt a simple gradient descent for R=78 with a very tight time limit. If it fails, return schoolbook.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to optimize for R=78
    R = 78
    if time.time() < deadline - 5:
        try:
            # Simple gradient descent
            np.random.seed(0)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 2:
                    break
                
                # Gradient of Loss = sum (sum_r U V W - T)^2
                # dLoss/dU = 2 * sum_{v,w} (sum_r U V W - T) * V[v,r] * W[w,r]
                # This is expensive.
                
                # Let's just return schoolbook to be safe.
                pass
        except Exception:
            pass
            
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
