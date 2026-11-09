"""Matrix multiplication 2x3x7 rank minimization.
Finds a lower-rank exact tensor decomposition if possible."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U and U[0] else 0
    # Build the tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k] -> C[i,k]
    # The convention is W[k*n+i, r] for the output
    # Tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta for the multiplication
    # Actually the tensor for matrix mult is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has entries T[i,j,k] = 1 (it's the identity in some sense)
    # Wait, let me reconsider. The matrix multiplication tensor T_{ijk} where i indexes rows of A, 
    # j indexes columns of A / rows of B, k indexes columns of B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    # Let me re-read the convention. The decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] is the matrix multiplication tensor.
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k? 
    # No wait. Let me think again.
    
    # Actually for matrix multiplication A (n x m) times B (m x p) giving C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid indices.
    # Hmm, that doesn't seem right for a tensor decomposition to be interesting.
    
    # Let me reconsider. The "matrix multiplication tensor" in the context of tensor rank
    # is defined as: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices correspond to
    # the multiplication. Actually T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # 
    # Wait no. Let me look at this more carefully.
    # The bilinear map (A, B) -> C where C = A*B.
    # In coordinates: C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor representing this is T_{ijk} where we contract over j.
    # T_{ijk} = 1 for all i,j,k.
    # 
    # So the tensor is just all ones! That seems too simple.
    # 
    # Actually I think I'm confusing myself. Let me reconsider.
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k
    # where the tensor product structure reflects how A, B, C are indexed.
    # So yes, all entries are 1.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U_arr[i*m + j]
                vj = V_arr[j*p + k]
                wk = W_arr[k*n + i]
                val = np.dot(ui, vj * wk)
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Try to find a lower rank decomposition
    # Known: rank of matrix multiplication tensor for (2,3,3) is 14 (Strassen-like)
    # For (2,3,7), we can potentially do better than 42.
    
    # Strategy: Use random initialization and gradient descent to minimize the error
    # of the tensor decomposition, then try to snap to integers/half-integers.
    
    # First, let's try some known small ranks
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Actually, the trivial lower bound is max(n*m, m*p, p*n) but that's not tight.
    # A better lower bound: rank >= n*m*p / min(n*m, m*p, p*n) ... no that's not right either.
    
    # Let's just try various ranks and see if we can find a solution.
    
    def try_rank(rank, max_iter=5000, lr=0.01, seed=42):
        """Try to find a decomposition of given rank."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient
            # Loss = sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
            # dLoss/dU[i*m+j,c] = 2 * (sum_c U[i*m+j,c'] V[j*p+k,c'] W[k*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
            # summed over k
            
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m + j
                        idx_v = j*p + k
                        idx_w = k*n + i
                        
                        prod = U[idx_u] * V[idx_v] * W[idx_w]
                        s = np.sum(prod)
                        diff = s - 1.0
                        loss += diff * diff
                        
                        # Gradient with respect to U[idx_u, c]
                        dU[idx_u] += 2 * diff * V[idx_v] * W[idx_w]
                        dV[idx_v] += 2 * diff * U[idx_u] * W[idx_w]
                        dW[idx_w] += 2 * diff * U[idx_u] * V[idx_v]
            
            # Gradient descent
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Adapt learning rate if loss is not decreasing
            if it > 100 and loss > best_loss * 1.1:
                lr *= 0.5
        
        return best_U, best_V, best_W, best_loss
    
    # Try ranks from low to high
    # Lower bound: need at least ceil(n*m*p / something)... 
    # Actually a simple lower bound: the tensor has rank at least the dimension of any "slice"
    # But let's just try from a reasonable lower bound.
    
    # For 2x3x7, let's try ranks 21 to 41
    # But this might take too long. Let's be strategic.
    
    # Try a few key ranks
    ranks_to_try = []
    for r in range(21, schoolbook_rank):
        ranks_to_try.append(r)
    
    # Sort by some heuristic - try ranks that divide nicely or are known to work for similar problems
    # For 2x3x3, rank 14 works (vs schoolbook 18)
    # For 2x3x7, maybe rank 34 or similar?
    
    # Let's prioritize: try 35, 34, 33, 32, 30, 28, 27, 25, 24, 23, 22, 21
    priority_ranks = [35, 34, 33, 32, 30, 28, 27, 25, 24, 23, 22, 21]
    # Add remaining
    for r in ranks_to_try:
        if r not in priority_ranks:
            priority_ranks.append(r)
    
    for rank in priority_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U, V, W, loss = try_rank(rank, max_iter=2000, lr=0.05, seed=seed)
            
            if loss < 1e-8:
                # Found a solution! Now try to snap to integers/half-integers
                # First check if it's already close to integers or half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Check if the snapped version still works
                U_list = [[float(x) for x in row] for row in U_int]
                V_list = [[float(x) for x in row] for row in V_int]
                W_list = [[float(x) for x in row] for row in W_int]
                
                if _check_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Try integer snapping
                    U_int2 = np.round(U)
                    V_int2 = np.round(V)
                    W_int2 = np.round(W)
                    
                    U_list2 = [[float(x) for x in row] for row in U_int2]
                    V_list2 = [[float(x) for x in row] for row in V_int2]
                    W_list2 = [[float(x) for x in row] for row in W_int2]
                    
                    if _check_decomposition(U_list2, V_list2, W_list2, n, m, p):
                        return {"U": U_list2, "V": V_list2, "W": W_list2}
                    else:
                        # Return the floating point solution (should be valid if loss is tiny)
                        U_list3 = [[float(x) for x in row] for row in U]
                        V_list3 = [[float(x) for x in row] for row in V]
                        W_list3 = [[float(x) for x in row] for row in W]
                        
                        if _check_decomposition(U_list3, V_list3, W_list3, n, m, p):
                            return {"U": U_list3, "V": V_list3, "W": W_list3}
    
    # Fall back to schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
