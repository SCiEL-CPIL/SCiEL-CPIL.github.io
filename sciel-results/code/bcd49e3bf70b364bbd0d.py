import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 2, 3, 9
    # p=9 is 3 blocks of p=3
    # We can solve for the rank of the 2x3x3 tensor and multiply by 3
    # Or solve for 2x3x9 directly if rank is low enough
    
    # Let's try to find the minimal rank for 2x3x3 first, then extend.
    # Actually, let's just solve for 2x3x9 directly using a greedy/randomized approach or a small rank solver.
    # The schoolbook rank is 54.
    # Known lower bounds: 
    # [2,2,2] rank 7
    # [2,3,3] rank is likely 12? Or 14?
    # Let's try to construct a valid decomposition for rank 36 (3 * 12) or lower.
    
    # Strategy:
    # 1. Define the tensor T of shape (n*m, m*p, p*n) which is 1 if the indices match the multiplication pattern, 0 otherwise.
    #    Actually, the tensor is defined such that T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    #    Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries 1 for all valid (i,j,k) in the range.
    #    The decomposition is: T = sum_{c=1}^R u_c \otimes v_c \otimes w_c
    #    where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    #    The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #    Wait, this is not quite right. The tensor entries are T[i,j,k] = 1 for all i,j,k.
    #    So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is a system of equations.
    # Let's try to find a low rank decomposition using random initialization and gradient descent?
    # Or use a known algebraic construction.
    
    # Let's try to construct a rank 36 decomposition (3 blocks of rank 12).
    # If rank 12 for 2x3x3 is not possible, we might need higher.
    # Let's try to solve for rank 12 for 2x3x3 using a solver if possible, or random search.
    
    # Given the time limit and complexity, let's try a simpler approach:
    # Use the fact that 9 = 3*3.
    # Maybe we can use a Kronecker product structure?
    # [2,3,9] = [2,3,3] x [1,1,3]? No.
    
    # Let's try to find a valid decomposition for rank 36.
    # If we assume the rank of 2x3x3 is 12, we can try to find it.
    # But if we can't, we can try rank 18 (schoolbook for 2x3x3) -> total 54.
    # We need to beat 54.
    
    # Let's try to find a rank 30 or 36 decomposition.
    # We will use a random search for the factors.
    
    # Define the target tensor T
    # T is of shape (6, 27, 18)
    # T[i,j,k] = 1 for all i,j,k?
    # No, the tensor is sparse in a sense? No, it's dense.
    # The condition is:
    # For all i in 0..1, j in 0..2, k in 0..8:
    # sum_{c=0}^{R-1} U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c] = 1
    
    # Let's try to find R such that a solution exists.
    # We can use a randomized algorithm:
    # 1. Initialize U, V, W with random Gaussian values.
    # 2. Use gradient descent to minimize sum (sum_c ... - 1)^2
    
    # This might be slow. Let's try a smaller rank first.
    # What is the known rank of 2x3x3?
    # It is known that rank(2,2,2)=7.
    # rank(2,3,3) is at most 18.
    # Is it 12?
    
    # Let's try to construct a rank 12 decomposition for 2x3x3 manually or via search.
    # If we fail, we return 54.
    
    # Actually, let's try to use the "border rank" or "subtensor" techniques?
    # No, we need exact rank.
    
    # Let's try a specific construction for 2x3x3 with rank 12.
    # If I can't find one, I'll try rank 14, 16, 18.
    
    # Let's try to use a solver for R=12.
    # The system is nonlinear.
    
    # Alternative: Use the fact that we have 120 seconds.
    # We can run a random search for R=30, 32, ... 54.
    # Start with R=30. If found, return.
    # If not, try R=32, etc.
    
    # But random search for tensor rank is hard.
    
    # Let's look at the structure again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For p=9, k = 3*b + l, where b in 0..2, l in 0..2.
    # C[i, 3b+l] = sum_j A[i,j] B[j, 3b+l]
    # This is exactly 3 independent 2x3x3 multiplications.
    # So rank(2,3,9) = 3 * rank(2,3,3).
    
    # So we just need to find the rank of 2x3x3.
    # Let's try to find a decomposition for 2x3x3 with rank R_block.
    # Total rank = 3 * R_block.
    
    # Let's try R_block = 12.
    # If that fails, R_block = 13, etc.
    
    # How to find R_block?
    # We can use a random search with restarts.
    
    def find_rank_233(R, time_budget):
        # Try to find U, V, W of shapes (6,R), (9,R), (6,R)
        # such that sum_c U[i*3+j,c] V[j*3+k,c] W[k*2+i,c] = 1 for all i,j,k
        
        # We can fix V to be identity-like or random and solve for U, W?
        # No, it's bilinear in U,W given V.
        
        # Let's use random initialization and gradient descent.
        # Loss = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
        
        # We can use Adam optimizer.
        
        # But implementing Adam from scratch is tedious.
        # Let's use a simpler approach: random search with local optimization.
        
        # Or, we can use the fact that we can fix V to be the identity matrix (if R >= 9)?
        # If V is identity, then V[j,k,c] = delta_{j,k} delta_{c,j}? No.
        
        # Let's try a specific algebraic construction.
        # For 2x2x2, rank 7 is known.
        # For 2x3x3, maybe rank 12 is achievable.
        
        # Let's try to construct it.
        # C00 = a00 b00 + a01 b10 + a02 b20
        # C01 = a00 b01 + a01 b11 + a02 b21
        # C02 = a00 b02 + a01 b12 + a02 b22
        # C10 = a10 b00 + a11 b10 + a12 b20
        # C11 = a10 b01 + a11 b11 + a12 b21
        # C12 = a10 b02 + a11 b12 + a12 b22
        
        # We need 18 products.
        # Can we reduce to 12?
        # Let's try to combine terms.
        
        # This is difficult to do by hand.
        # Let's try a computational search.
        
        # We will use a simple random search.
        # Initialize U, V, W with small random numbers.
        # Use a few iterations of gradient descent.
        
        # Since we are in Python, we can use numpy for vectorization.
        
        # Let's try R=12.
        np.random.seed(42)
        
        U = np.random.randn(6, R) * 0.1
        V = np.random.randn(9, R) * 0.1
        W = np.random.randn(6, R) * 0.1
        
        # Target tensor T of shape (2,3,3) with all 1s.
        # We want to minimize || T - sum_c U_c \otimes V_c \otimes W_c ||^2
        
        # We can compute the current tensor and the error.
        
        # Let's do 1000 iterations of gradient descent with step size 0.01.
        
        # This might be too slow in pure Python/NumPy for 120s.
        # But R=12 is small.
        
        # Let's try.
        
        for it in range(1000):
            if time.time() > start_time + time_budget:
                return None
            
            # Compute tensor
            # T[i,j,k] = sum_c U[i*3+j,c] V[j*3+k,c] W[k*2+i,c]
            
            # This is a bit complex to vectorize efficiently without loops over i,j,k.
            # But 2*3*3 = 18 terms. We can just loop.
            
            T = np.zeros((2,3,3))
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        T[i,j,k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*2+i, :])
            
            # Error
            E = T - 1.0
            
            # Gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        dU[i*3+j, :] += E[i,j,k] * V[j*3+k, :] * W[k*2+i, :]
                        dV[j*3+k, :] += E[i,j,k] * U[i*3+j, :] * W[k*2+i, :]
                        dW[k*2+i, :] += E[i,j,k] * U[i*3+j, :] * V[j*3+k, :]
            
            # Update
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if np.max(np.abs(E)) < 1e-6:
                return U, V, W
        
        return None
    
    # Try to find rank 12 for 2x3x3
    # If successful, total rank is 36.
    
    U_block, V_block, W_block = None, None, None
    
    # Try R=12
    if time.time() < deadline - 10:
        res = find_rank_233(12, 10)
        if res is not None:
            U_block, V_block, W_block = res
    
    # If not found, try R=14
    if U_block is None and time.time() < deadline - 10:
        res = find_rank_233(14, 10)
        if res is not None:
            U_block, V_block, W_block = res
    
    # If not found, try R=16
    if U_block is None and time.time() < deadline - 10:
        res = find_rank_233(16, 10)
        if res is not None:
            U_block, V_block, W_block = res
    
    # If not found, use schoolbook rank 18 for block (total 54)
    if U_block is None:
        R_block = 18
        U_block = np.zeros((6, R_block))
        V_block = np.zeros((9, R_block))
        W_block = np.zeros((6, R_block))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_block[i*3+j, col] = 1
                    V_block[j*3+k, col] = 1
                    W_block[k*2+i, col] = 1
                    col += 1
    
    # Now construct the full 2x3x9 decomposition
    # Total rank = 3 * R_block
    R_total = 3 * R_block if R_block != 18 else 54 # If R_block was 18, total is 54.
    # Actually, if R_block was found to be 12, 14, or 16, total is 3*R_block.
    # If R_block was 18 (schoolbook), total is 54.
    
    # Wait, if R_block is 12, total is 36.
    # If R_block is 14, total is 42.
    # If R_block is 16, total is 48.
    # If R_block is 18, total is 54.
    
    # We need to map the block indices to the full indices.
    # B is 3x9. B[j, 3b+l] is the (j,l) element of the b-th block.
    # So V_full[j*9 + 3b + l, c_block + b*R_block] = V_block[j*3 + l, c_block]
    # W_full[(3b+l)*2 + i, c_block + b*R_block] = W_block[l*2 + i, c_block]
    # U_full[i*3 + j, c_block + b*R_block] = U_block[i*3 + j, c_block]
    
    U = np.zeros((6, R_total))
    V = np.zeros((27, R_total))
    W = np.zeros((18, R_total))
    
    for b in range(3):
        for c in range(R_block):
            c_full = b * R_block + c
            # U
            U[:, c_full] = U_block[:, c]
            # V
            for j in range(3):
                for l in range(3):
                    V[j*9 + 3*b + l, c_full] = V_block[j*3 + l, c]
            # W
            for l in range(3):
                for i in range(2):
                    W[(3*b + l)*2 + i, c_full] = W_block[l*2 + i, c]
    
    # Verify the solution
    # Check a few random points
    # A = np.random.randn(2,3)
    # B = np.random.randn(3,9)
    # C = A @ B
    # C_calc = np.zeros((2,9))
    # for i in range(2):
    #     for k in range(9):
    #         C_calc[i,k] = np.sum(U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c] for j in range(3) for c in range(R_total))
    #         # This is O(R * 3) per element, so O(9*2*3*R) = 54R operations.
    #         # For R=36, that's 1944 operations. Very fast.
    
    # Let's do a full verification to be sure.
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 9)
    C_expected = A @ B
    C_calc = np.zeros((2, 9))
    
    # Efficient verification:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our formula: C[i,k] = sum_c U[i*3+j, c] V[j*9+k, c] W[k*2+i, c]
    # This is not directly A[i,j]B[j,k].
    # The tensor decomposition satisfies:
    # sum_c U[i*3+j, c] V[j*9+k, c] W[k*2+i, c] = 1 for all i,j,k?
    # No, the tensor T[i,j,k] = 1.
    # The multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says: T = sum_c u_c v_c w_c
    # Where T[i,j,k] = 1.
    # Then <T, A \otimes B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # This is not C. C is a 2x9 matrix.
    # The contraction is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that the bilinear form <T, A \otimes B> gives the product?
    # No, the standard definition is:
    # T = sum_{i,j,k} e_i \otimes f_j \otimes g_k
    # Then <T, A \otimes B> = sum_{i,j,k} A[i,j] B[j,k] g_k? No.
    
    # Let's re-read the task description.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The tensor is T such that T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Since every A[i,j]B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T = sum_c U_c \otimes V_c \otimes W_c
    # So sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    # If this holds, then:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # Multiply by A[i,j] B[j,k] and sum over j?
    # No, the contraction is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want to express C[i,k] as sum_c (sum_j A[i,j] U[i*m+j, c]) (sum_j B[j,k] V[j*p+k, c]) W[k*n+i, c]
    # Let X_c[i] = sum_j A[i,j] U[i*m+j, c]
    # Let Y_c[k] = sum_j B[j,k] V[j*p+k, c]
    # Then C[i,k] = sum_c X_c[i] Y_c[k] W[k*n+i, c]
    
    # This requires:
    # sum_c (sum_j A[i,j] U[i*m+j, c]) (sum_j B[j,k] V[j*p+k, c]) W[k*n+i, c] = sum_j A[i,j] B[j,k]
    
    # This is true if:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{j,j'}? No.
    # It is true if the tensor T[i,j,k] = 1 is decomposed as sum_c U_c V_c W_c.
    # Then:
    # sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c])?
    # No, the W index is fixed for the output C[i,k].
    
    # The correct identity is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c])
    
    # This works if:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    # Wait, if I sum over j in the expression above, I get:
    # sum_j A[i,j] B[j,k] sum_c ...
    # This is not 1.
    
    # The tensor T[i,j,k] is 1.
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # This means T[i,j,k] = sum_c u_c[i] v_c[j] w_c[k].
    # Here u_c is indexed by i (row of A), v_c by j (col of A / row of B), w_c by k (col of B).
    # But U has size n*m. So U[i*m+j, c] is u_c[i,j]?
    # No, the standard tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i \otimes f_j \otimes g_k
    # The factors are:
    # U: (n*m) x R, indexed by (i,j)
    # V: (m*p) x R, indexed by (j,k)
    # W: (p*n) x R, indexed by (k,i)
    
    # The condition is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    # If this holds, then:
    # sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * 1
    # = sum_j A[i,j] B[j,k] (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c])
    # = sum_c W[k*n+i, c] (sum_j A[i,j] U[i*m+j, c]) (sum_j B[j,k] V[j*p+k, c])
    
    # Yes! This is correct.
    
    # So we just need to verify:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's verify our constructed U, V, W.
    
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                val = np.sum(U[i*3+j, :] * V[j*9+k, :] * W[k*2+i, :])
                if abs(val - 1.0) > 1e-5:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook 54
        rank = 54
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    U[i*3+j, col] = 1
                    V[j*9+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
