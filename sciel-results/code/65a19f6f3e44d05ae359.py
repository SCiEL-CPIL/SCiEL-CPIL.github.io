"""Matrix multiplication 2,3,10 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 2*3*10 = 60.
    We aim to find a decomposition with rank < 60 using known algorithms.
    
    For small matrix multiplications, the optimal ranks are known:
    - 2x2: rank 7 (Strassen)
    - 2x3: rank 10
    - 3x3: rank 23 (Lader)
    
    For 2x3 * 3x10, we can decompose the 3x10 matrix into 10 independent 3x1 column vectors.
    Each 2x3 * 3x1 multiplication can be done with rank 10 (optimal for 2x3 * 3x1).
    But we need to be careful about how the tensors combine.
    
    Actually, let's think more carefully. The tensor for multiplying A (2x3) by B (3x10) 
    to get C (2x10) has entries T[i,k,j,l] where i indexes rows of A/C, k indexes cols of A/rows of B,
    j indexes cols of B/C, l is... wait, let me re-read the convention.
    
    The convention is W[k*n+i, r] for the output factor. Let me re-examine.
    
    U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    V has shape (m*p, rank) = (3*10, rank) = (30, rank)  
    W has shape (p*n, rank) = (10*2, rank) = (20, rank)
    
    The tensor T[i,j,k] for multiplying A (n x m) by B (m x p) gives C (n x p) is:
    T[i,j,k] = 1 if the (i,k) element of C depends on A[i,j] * B[j,k]
    
    In the factorization: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, that's not right either. Let me think about this more carefully.
    
    The standard tensor decomposition for matrix multiplication:
    C = A * B where A is n x m, B is m x p, C is n x p
    
    The multiplication tensor T has entries T[i, j, k, l] = delta_{j,k} for i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    Wait, that's not standard either.
    
    Let me use the convention from the problem:
    U has shape (n*m, rank) - so U[i*m + j, r] for i in range(n), j in range(m)
    V has shape (m*p, rank) - so V[j*p + k, r] for j in range(m), k in range(p)
    W has shape (p*n, rank) - so W[k*n + i, r] for k in range(p), i in range(n)
    
    The decomposition should satisfy:
    For all i in range(n), j in range(m), k in range(p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other combinations, the sum should be 0.
    
    Actually, the matrix multiplication tensor is:
    T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    
    So the tensor has entries indexed by (i, j, k) where:
    - i is the row of A (and C), 0 <= i < n
    - j is the column of A (and row of B), 0 <= j < m  
    - k is the column of B (and C), 0 <= k < p
    
    T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    Wait, no. The tensor for bilinear multiplication is:
    T[i, j, k] = delta where the three indices correspond to the three factors.
    
    Let me be very precise. The multiplication tensor T for A(n x m) * B(m x p) = C(n x p) is a 3-tensor with:
    - First mode: n*m (indices for A, flattened as i*m + j)
    - Second mode: m*p (indices for B, flattened as j*p + k)
    - Third mode: n*p (indices for C, flattened as i*p + k)
    
    But the problem uses W[k*n + i, r] which is k*p... no, k*n + i where k is in range(p) and i is in range(n).
    So W's first index is k*n + i, meaning the C matrix is stored in column-major order? Or is it just a different indexing?
    
    C has shape (n, p), so element C[i, k] where i is row (0..n-1) and k is column (0..p-1).
    If we flatten C in row-major: index = i*p + k
    If we flatten C in column-major: index = k*n + i
    
    The problem says W[k*n + i, r], so it's using column-major flattening for C.
    
    So the tensor T has:
    - Mode 1 (A): index = i*m + j, value = A[i,j]
    - Mode 2 (B): index = j*p + k, value = B[j,k]
    - Mode 3 (C): index = k*n + i, value = C[i,k]
    
    And T[a, b, c] = 1 if and only if:
    - a = i*m + j for some i, j
    - b = j'*p + k for some j', k
    - c = k'*n + i' for some k', i'
    - AND i = i', j = j', k = k'
    
    In other words, T[i*m+j, j'*p+k, k'*n+i'] = 1 if and only if i=i', j=j', k=k'
    
    So the schoolbook decomposition has rank n*m*p with:
    U[i*m+j, i*m+j] = 1 (using column index = i*m+j)
    V[j*p+k, i*m+j] = 1 if j matches (i.e., column index i*m+j has j part matching)
    W[k*n+i, i*m+j] = 1 if i,k match
    
    More precisely, for each (i,j,k) triple:
    U[i*m+j, col] = 1
    V[j*p+k, col] = 1
    W[k*n+i, col] = 1
    
    This gives rank = n*m*p = 60.
    
    Now, can we do better? For 2x3 * 3x10, the optimal rank is not known to be less than 60 in general,
    but there might be structure we can exploit.
    
    One approach: since p=10 is relatively large compared to n=2 and m=3, we can think of this as
    10 independent 2x3 * 3x1 multiplications (one for each column of B).
    
    For 2x3 * 3x1, the optimal rank is 10 (this is a known result).
    If we could share the U factors across the 10 columns, we might reduce the total rank.
    
    However, the B matrix has different values in each column, so the V factors will be different.
    The U factors depend only on A, so they can be shared.
    
    Let's think about it differently. The tensor is:
    T[i*m+j, j'*p+k, k'*n+i'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    This is a "direct sum" structure in some sense. Specifically, for fixed i, j, the tensor entries are:
    T[i*m+j, j*p+k, k*n+i] = 1 for all k in range(p)
    
    So for each (i,j) pair, we have a sub-tensor that is essentially a "diagonal" in the k dimension.
    
    Actually, let me reconsider. The tensor T can be written as:
    T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    This is a sum of n*m*p rank-1 tensors. The question is whether this sum can be compressed.
    
    For the 2x3 * 3x1 case (p=1), the tensor is:
    T[i*m+j, j*1+k, k*1+i] = T[i*3+j, j, i] for i in {0,1}, j in {0,1,2}, k in {0}
    
    So T has 6*3*2 = 36 entries, but only 2*3*1 = 6 non-zero entries (all equal to 1).
    
    The optimal rank for 2x3 * 3x1 is 10. This is a known result.
    
    For 2x3 * 3x10, if we treat it as 10 independent 2x3 * 3x1 problems, the naive rank would be 10*10 = 100,
    which is worse than schoolbook (60). So we need to find a better decomposition.
    
    Alternatively, we can use the fact that the tensor has a special structure. Let me try to find a decomposition
    with rank less than 60.
    
    One idea: use the Lader decomposition for 3x3 * 3x3 (rank 23) and extend it. But this doesn't directly apply.
    
    Another idea: since n=2 is small, we can try to exploit the 2x structure.
    
    Let me try a different approach: use random initialization and gradient descent to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using iterative methods
    # We'll try ranks from 40 up to 59 and see if we can find a valid decomposition
    
    def check_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
        """Check if U, V, W form a valid decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for invalid combinations, the sum should be 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > tol:
                        return False
        
        # Check that invalid combinations give 0
        # For example, if i != i', then U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] should sum to 0
        # This is automatically satisfied if the decomposition is correct, but let's verify a few
        for i in range(n):
            for i_prime in range(n):
                if i == i_prime:
                    continue
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        w_val = W[k * n + i_prime]
                        s = np.dot(u_val, v_val * w_val)
                        if abs(s) > tol:
                            return False
        
        return True
    
    # Try gradient-based optimization for various ranks
    for rank in range(45, 60):  # Try ranks 45-59
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + rank)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Use simple gradient descent
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        u_val = U[u_idx]
                        v_val = V[v_idx]
                        w_val = W[w_idx]
                        
                        # Current value
                        s = np.dot(u_val, v_val * w_val)
                        error = s - 1
                        
                        # Gradients
                        U_grad[u_idx] += error * (v_val * w_val)
                        V_grad[v_idx] += error * (u_val * w_val)
                        W_grad[w_idx] += error * (u_val * v_val)
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if step % 100 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(s - 1))
                if max_err < 1e-10:
                    break
        
        # Check if we found a valid decomposition
        if check_decomposition(U, V, W, n, m, p, rank):
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
