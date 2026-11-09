<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify exact decomposition by checking random matrix products."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        np.random.seed(12345)
        for trial in range(8):
            A = np.random.randint(-5, 6, size=(n, m)).astype(np.float64)
            B = np.random.randint(-5, 6, size=(m, p)).astype(np.float64)
            C_true = A @ B
            
            C_decomp = np.zeros((n, p), dtype=np.float64)
            for r in range(rank):
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for j in range(m):
                            val += U_arr[i * m + j, r] * A[i, j] * V_arr[j * p + k, r] * B[j, k]
                        C_decomp[i, k] += val * W_arr[k * n + i, r]
            
            if not np.allclose(C_decomp, C_true, atol=tol, rtol=tol):
                return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Strategy: Use Strassen's 7-term decomposition for 2x2 blocks
    # to build a structured decomposition for 4x5 * 5x5
    
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x5 * 5x5, decompose as blocks:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # This is getting complex. Let me try a different approach:
    # Directly search for lower-rank decompositions using the structure
    
    # Try to find a rank-r decomposition by solving a system of equations
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms
    
    # Alternative: Use the fact that the all-ones tensor of size n*m*p has rank
    # that can be bounded using known results
    
    # For now, let's try random search with structured initialization
    # focusing on the lower bound max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    
    # Try ranks from 25 upward, with increasing effort for lower ranks
    for target_rank in range(25, min(best_rank, 60)):
        if time.time() > deadline - 15:
            break
        
        # Number of attempts based on rank (more attempts for lower ranks)
        attempts = max(1, 10 - (target_rank - 25) // 5)
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Structured random initialization
            # Use values in {-1, 0, 1} with some structure
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Ensure at least some non-zero entries in each row
            for r in range(target_rank):
                if np.all(U[:, r] == 0):
                    U[np.random.randint(n * m), r] = 1
                if np.all(V[:, r] == 0):
                    V[np.random.randint(m * p), r] = 1
                if np.all(W[:, r] == 0):
                    W[np.random.randint(p * n), r] = 1
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                if is_valid_entries(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best_result
        
        if time.time() > deadline - 25:
            break
    
    # Try a more systematic approach: use Kronecker product structure
    # Embed 4x5*5x5 into a larger problem and use known decompositions
    
    # For 2x2*2x2, Strassen gives rank 7
    # For 4x4*4x4, we can use Strassen recursively: rank <= 7^2 = 49
    # But our problem is 4x5*5x5, not square
    
    # Let's try to use the 2x2 Strassen decomposition as a building block
    # and extend it to our dimensions
    
    # Create a decomposition using Strassen-like structure
    # This is heuristic and may not give optimal rank, but let's try
    
    if time.time() < deadline - 30:
        # Try a structured decomposition based on block Strassen
        # Split 4x5 into 2x2 and 2x3 blocks, 5x5 into 2x2, 2x3, 3x2, 3x3
        
        # For simplicity, let's try a direct construction
        # Use the schoolbook but try to find linear dependencies
        
        # Check if any subset of schoolbook terms is linearly dependent
        # This is a hard problem, so let's skip for now
        
        # Final attempt: try random search with half-integer entries
        for target_rank in range(25, min(best_rank, 50)):
            if time.time() > deadline - 20:
                break
            
            attempts = max(1, 5 - (target_rank - 25) // 10)
            for attempt in range(attempts):
                if time.time() > deadline - 15:
                    break
                
                # Use half-integer entries
                U = np.random.choice([-0.5, 0, 0.5, 1], size=(n * m, target_rank))
                V = np.random.choice([-0.5, 0, 0.5, 1], size=(m * p, target_rank))
                W = np.random.choice([-0.5, 0, 0.5, 1], size=(p * n, target_rank))
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify_decomposition(U_list, V_list, W_list):
                    if is_valid_entries(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition_fast(U_arr, V_arr, W_arr, tol=1e-10, n_trials=5):
        """Verify exact decomposition using vectorized operations."""
        rank = U_arr.shape[1]
        
        # Precompute for faster verification
        # For each r, we need: sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # This is equivalent to: for each (i,k), C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j,r]*A[i,j])*(V[j*p+k,r]*B[j,k])
        
        np.random.seed(42)
        for trial in range(n_trials):
            A = np.random.randint(-3, 4, size=(n, m)).astype(np.float64)
            B = np.random.randint(-3, 4, size=(m, p)).astype(np.float64)
            C_true = A @ B
            
            C_decomp = np.zeros((n, p), dtype=np.float64)
            
            for r in range(rank):
                # u_r[i,j] = U[i*m+j, r]
                # v_r[j,k] = V[j*p+k, r]
                # w_r[k,i] = W[k*n+i, r]
                
                u_r = U_arr[:, r].reshape(n, m)
                v_r = V_arr[:, r].reshape(m, p)
                w_r = W_arr[:, r].reshape(p, n)
                
                # Compute: sum_j u_r[i,j]*A[i,j] * v_r[j,k]*B[j,k] * w_r[k,i]
                # = sum_i,k w_r[k,i] * sum_j (u_r[i,j]*A[i,j]) * (v_r[j,k]*B[j,k])
                
                # ua[i,j] = u_r[i,j] * A[i,j]
                ua = u_r * A
                # vb[j,k] = v_r[j,k] * B[j,k]
                vb = v_r * B
                # intermediate[i,k] = sum_j ua[i,j] * vb[j,k]
                intermediate = ua @ vb
                # C_decomp[i,k] += w_r[k,i] * intermediate[i,k]
                C_decomp += w_r.T * intermediate
            
            if not np.allclose(C_decomp, C_true, atol=tol, rtol=tol):
                return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Strategy: Use structured decompositions based on known results
    # For 4x5 * 5x5, we can try to embed into larger square problems
    
    # First, try a block-based approach using Strassen for 2x2 blocks
    # A is 4x5, B is 5x5
    # Split A into 2x2 blocks: A11(2x2), A12(2x3), A21(2x2), A22(2x3)
    # Split B into 2x2 blocks: B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    
    # Actually, let's use a simpler approach: try to find a decomposition
    # by solving for the tensor structure
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 3-dimensional tensor of all ones with shape (n*m, m*p, p*n)
    # but with specific index mappings
    
    # Let's try a more systematic search using the structure
    
    # Approach 1: Try ranks from a reasonable lower bound
    # The rank of matrix multiplication tensor is at least max(n*m, m*p, p*n)
    # For 4x5*5x5: max(20, 25, 20) = 25
    # But known results suggest the rank is much higher
    
    # Let's try to construct a valid decomposition using a combination of
    # schoolbook terms with possible cancellations
    
    # For now, let's do a focused search with better verification
    
    # Try to find decompositions with rank in a reasonable range
    # Based on known results, the rank for 4x4*4x4 is at most 49 (Strassen^2)
    # For 4x5*5x5, we can embed into 5x5*5x5 which has rank at most 7^2=49 for the 4x4 part
    
    # Let's try a direct construction approach
    
    # Method: Use the fact that we can write the multiplication as a sum of
    # rank-1 tensors, and try to find a sparse representation
    
    # For efficiency, let's try ranks from 40 to 100 with structured random search
    
    for target_rank in range(40, min(best_rank, 100)):
        if time.time() > deadline - 20:
            break
        
        # Number of attempts: more for lower ranks
        if target_rank <= 50:
            attempts = 20
        elif target_rank <= 60:
            attempts = 10
        elif target_rank <= 80:
            attempts = 5
        else:
            attempts = 3
        
        for attempt in range(attempts):
            if time.time() > deadline - 15:
                break
            
            # Use structured random initialization with {-1, 0, 1}
            # Ensure sparsity to make verification more likely to succeed
            density = max(0.1, 0.5 - target_rank / 200)
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                # Randomly select which entries are non-zero
                u_mask = np.random.rand(n * m) < density
                v_mask = np.random.rand(m * p) < density
                w_mask = np.random.rand(p * n) < density
                
                # Ensure at least one non-zero in each
                if not np.any(u_mask):
                    u_mask[np.random.randint(n * m)] = True
                if not np.any(v_mask):
                    v_mask[np.random.randint(m * p)] = True
                if not np.any(w_mask):
                    w_mask[np.random.randint(p * n)] = True
                
                U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
                V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
                W[w_mask, r] = np.random.choice([-1, 1], size=np.sum(w_mask))
            
            if verify_decomposition_fast(U, V, W, n_trials=3):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if is_valid_entries(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best_result
    
    # If we haven't found anything better, try a different strategy
    # Use a more systematic construction based on block decomposition
    
    if time.time() < deadline - 30:
        # Try to construct a decomposition using 2x2 Strassen blocks
        # Split the 4x5 matrix into blocks and use Strassen's algorithm
        
        # For simplicity, let's try to use the schoolbook but with some
        # clever grouping to reduce rank
        
        # Actually, let's try a different approach: use the fact that
        # we can factor the all-ones tensor in various ways
        
        # The tensor T[i,j,k] = 1 can be written as:
        # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
        # where u_r, v_r, w_r are vectors
        
        # For the matrix multiplication tensor, we have:
        # T[i,j,k] = 1 for all i,j,k (in the appropriate indexing)
        
        # Let's try to find a low-rank decomposition of the all-ones tensor
        # with the specific index structure
        
        # For a tensor of all ones with shape (a,b,c), the rank is min(a,b,c)
        # if we can find appropriate factorizations
        
        # In our case, the tensor has shape (20, 25, 20) with specific index mappings
        # The rank is at least 20 (the minimum dimension)
        
        # Let's try to construct such a decomposition
        
        # For the all-ones tensor of shape (a,b,c) with rank r:
        # U[i,r], V[j,r], W[k,r] such that sum_r U[i,r]*V[j,r]*W[k,r] = 1
        
        # One way: if r = min(a,b,c), we can use:
        # U[i,r] = 1 if i < r else 0, V[j,r] = 1 if j < r else 0, W[k,r] = 1 if k < r else 0
        # But this only works if a=b=c=r
        
        # For our case, we need a more sophisticated approach
        
        # Let's try: U[i,r] = 1 for all i,r, V[j,r] = 1 for all j,r, W[k,r] = 1/r for all k,r
        # Then sum_r U[i,r]*V[j,r]*W[k,r] = sum_r 1*1*(1/r) = r*(1/r) = 1
        # This works! But the rank is r, and we need to check if this is valid
        
        # Wait, this gives a rank-r decomposition for any r!
        # But we need to check if the entries are valid (integer or half-integer)
        # 1/r is only integer or half-integer for r = 1, 2
        
        # For r=1: U[i,0]=1, V[j,0]=1, W[k,0]=1 -> rank 1, but this doesn't work
        # because the index structure is more complex
        
        # The issue is that our tensor isn't just all ones in the (i,j,k) space
        # It's indexed as U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # Let me reconsider the structure...
        
        # For the matrix multiplication tensor:
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # We need: (AB)[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k])
        
        # This is equivalent to: for each (i,k), the coefficient of A[i,j]*B[j,k] in the
        # expansion must be 1 for each j, and 0 otherwise
        
        # So we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k
        
        # This is a system of n*m*p = 100 equations
        
        # Let's try to solve this system for small r
        
        # For r=1: W[k*n+i,0] * U[i*m+j,0] * V[j*p+k,0] = 1 for all i,j,k
        # This implies U, V, W are all 1 (up to sign), and W*U*V = 1
        # So U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1 works!
        # But wait, this gives rank 1, which seems too good to be true
        
        # Let me verify: if U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
        # Then sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1*1*1 = 1
        # This works for all i,j,k!
        
        # But this can't be right because the rank of matrix multiplication is known
        # to be at least max(n*m, m*p, p*n) = 25 for 4x5*5x5
        
        # The issue is that I'm misunderstanding the tensor structure
        
        # Let me re-read the problem statement...
        
        # The decomposition is:
        # C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
        
        # For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
        # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta[j,j'] for each (i,k)
        # Wait, that's not right either
        
        # Actually, for the decomposition to work for ALL matrices A and B, we need:
        # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k
        
        # Because the coefficient of A[i,j]*B[j,k] in the expansion must be 1
        
        # So if we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1/r for all i,j,k,r
        # Then sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = sum_r (1/r)*1*1 = r*(1/r) = 1
        
        # This works! The rank is r, and we can choose any r >= 1
        
        # But the entries must be integer or half-integer
        # 1/r is integer or half-integer only for r = 1 or r = 2
        
        # For r=1: W[k*n+i, 0] = 1, which is integer
        # For r=2: W[k*n+i, r] = 1/2, which is half-integer
        
        # Let's try r=1 first:
        # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
        # This should work!
        
        # But wait, this gives rank 1, which is impossible for matrix multiplication
        # The rank of matrix multiplication is known to be much higher
        
        # I think I'm making an error in my understanding
        
        # Let me think about this more carefully...
        
        # The matrix multiplication tensor is defined such that:
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        
        # A rank-r decomposition gives:
        # (AB)[i,k] = sum_{r=1}^R sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        
        # For this to hold for all A and B, we need:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
        
        # This is a system of n*m*p equations
        
        # If we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1/R for all i,j,k,r
        # Then sum_r (1/R)*1*1 = R*(1/R) = 1
        # This works!
        
        # So the rank is R, and we can choose any R >= 1
        
        # But the entries must be integer or half-integer
        # 1/R is integer or half-integer only for R = 1 or R = 2
        
        # For R=1: all entries are 1, which is integer
        # For R=2: W entries are 1/2, which is half-integer
        
        # Let's try R=1:
        U1 = [[1] * 1 for _ in range(n * m)]
        V1 = [[1] * 1 for _ in range(m * p)]
        W1 = [[1] * 1 for _ in range(p * n)]
        
        if verify_decomposition_fast(np.array(U1), np.array(V1), np.array(W1), n_trials=3):
            best_result = {"U": U1, "V": V1, "W": W1}
            best_rank = 1
            return best_result
        
        # Try R=2:
        U2 = [[1, 1] for _ in range(n * m)]
        V2 = [[1, 1] for _ in range(m * p)]
        W2 = [[0.5, 0.5] for _ in range(p * n)]
        
        if verify_decomposition_fast(np.array(U2), np.array(V2), np.array(W2), n_trials=3):
            best_result = {"U": U2, "V": V2, "W": W2}
            best_rank = 2
            return best_result
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in my reasoning. Let me reconsider the problem.

The issue is that my simple construction doesn't account for the fact that the tensor structure is more complex. Let me fix the program with a correct approach:

