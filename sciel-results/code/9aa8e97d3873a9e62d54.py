"""Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 145s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 60 as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random search with integer/half-integer entries
    # The matrix multiplication tensor T_{ijk} = delta_{i,i}delta_{j,j}delta_{k,k} in the sense that
    # (A*B)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k, 0 otherwise
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for each column c, U[:,c] * V[:,c] * W[:,c] should sum to the multiplication tensor
    # The multiplication tensor M has entries M_{(i,j),(j,k),(k,i)} = 1 if the indices are consistent
    # Actually, the standard convention: the multiplication tensor for C = A*B where A is n×m, B is m×p
    # has entries T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n
    # The entry (i*m+j, j*p+k, k*n+i) of the tensor product should match
    # Actually, let me think about this more carefully.
    
    # The tensor for matrix multiplication A(n×m) * B(m×p) = C(n×p) is:
    # T_{i,j,k} = 1 for all valid i,j,k
    # The decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    # The index (i*m+j) in u corresponds to (i,j), (j*p+k) in v corresponds to (j,k), (k*n+i) in w corresponds to (k,i)
    # For the tensor to be correct, we need:
    # sum_c u_{i*m+j} * v_{j*p+k} * w_{k*n+i} = 1 for all i,j,k
    # and = 0 for inconsistent indices (but with this indexing, all indices are consistent)
    
    # So we need: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1 for all i,j,k
    
    # Let's use a random search approach with small rank
    # Start with rank 40 and try to find a solution
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
        """Fast verification using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # U_arr shape: (n*m, rank)
        # V_arr shape: (m*p, rank)
        # W_arr shape: (p*n, rank)
        # For each c, we have outer product U_arr[:,c] ⊗ V_arr[:,c] ⊗ W_arr[:,c]
        # We need sum over c of this to equal the multiplication tensor
        # The multiplication tensor has entries 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k
        # Let's compute the sum
        # sum_c U[:,c] V[:,c] W[:,c] is a 3D tensor of shape (n*m, m*p, p*n)
        # We can compute this as: for each c, outer(U[:,c], outer(V[:,c], W[:,c]))
        # But this is expensive. Instead, let's just check the relevant entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from 40 down to 30
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fallback
    
    # Use a structured random search
    # The entries should be integers or half-integers (multiples of 0.5)
    # Let's try a random search with entries in {-1, -0.5, 0, 0.5, 1}
    
    random.seed(42)
    
    for target_rank in [40, 38, 36, 34, 32, 30]:
        if time.time() > deadline - 30:
            break
        
        found = False
        num_trials = 0
        max_trials = 20000
        
        while not found and num_trials < max_trials and time.time() < deadline - 10:
            num_trials += 1
            
            # Initialize with random values
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Quick check: compute the sum for each (i,j,k)
            # Use numpy for speed
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        if abs(s - 1.0) > 1e-4:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                found = True
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
            elif num_trials % 1000 == 0:
                # Could do local refinement here, but keep it simple for now
                pass
    
    if best_rank < rank_fallback:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
