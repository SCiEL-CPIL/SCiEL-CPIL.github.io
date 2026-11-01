"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition search."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U @ W.T @ V.T reproduces the matmul tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need: for each (i,j,k), sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] == 1
    # The matmul tensor T has T[i,j,k] = delta (i.e., the product C_ik = sum_j A_ij B_jk)
    # Actually the tensor is T[i,j,k] where (i,j) is input to A, (j,k) input to B, output (i,k)
    # The tensor for matmul: T[i,j,k] = 1 if and only if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = delta(i,j,k) in the sense that each term A_ij * B_jk contributes to C_ik
    # The decomposition: T = sum_c U_c ⊗ W_c ⊗ V_c where
    #   U_c[i*m+j] * W_c[k*n+i] * V_c[j*p+k] = T[i,j,k]
    # T[i,j,k] = 1 for all (i,j,k) in range(n) x range(m) x range(p) -- no wait
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if we're at position (i,j,k) meaning A[i,j] * B[j,k]
    # This is just the identity in a sense -- every (i,j,k) triple contributes exactly 1
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let's verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * W_arr[k*n+i, :] * V_arr[j*p+k, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100.0  # 100s budget, leave 20s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Strategy: try to find a decomposition with rank < 48
    # For 2x3xn matrices, known results suggest rank can be significantly less than 6n.
    # Let's try a structured search for low-rank decompositions.
    
    # Key insight: For M_{2,3,8}, we can try ranks from low to high.
    # The tensor has structure: T[i,j,k] = 1 for all (i,j,k).
    # A rank-r decomposition means: for all (i,j,k), sum_{c=1}^{r} U[i*m+j,c]*W[k*n+i,c]*V[j*p+k,c] = 1
    
    # Let's try a randomized approach: for each candidate rank, randomly initialize
    # and use iterative refinement to satisfy the constraints.
    
    rng = np.random.RandomState(42)
    
    # Try various ranks from low to high
    # Lower bound: at least n*p = 16 (since output is 2x8=16 entries)
    # Also need at least m = 3 (inner dimension)
    # Known: for 2x2xn, rank ~ 2n-1 or so. For 2x3x8, maybe ~ 2*8 + something.
    
    # Let's try a more systematic approach:
    # The tensor T[i,j,k] = 1. We need U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] to sum to 1.
    
    # One approach: fix W (or U) and solve for the others.
    # If we fix V and W, then for each c, U[:,c] is determined by:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    # This is linear in U if V and W are fixed.
    
    # Let's try: parameterize and use least squares / linear solve.
    
    # Actually, let's try a different approach: use the fact that the tensor is simple.
    # T[i,j,k] = 1 means we can write it as sum over c of rank-1 tensors.
    
    # For rank r, we need:
    # For each (i,j,k): sum_c a_{i,j,c} * b_{k,i,c} * d_{j,k,c} = 1
    # where a = U reshaped, b = W reshaped, d = V reshaped.
    
    # Let's try small ranks with structured initialization.
    
    # Try rank from 20 to 47
    for rank in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U_init = rng.randn(n*m, rank) * 0.5
            V_init = rng.randn(m*p, rank) * 0.5
            W_init = rng.randn(p*n, rank) * 0.5
            
            # Iterative refinement: alternate between solving for one factor
            # while keeping the others fixed.
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U: for each (i,j), sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for all k
                # This gives us equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c]*W[k*n+i,c] = 1
                # Group by (i,j): for fixed (i,j), we have p equations in rank unknowns U[i*m+j, :]
                # A[i*m+j, c] = V[j*p+k,c] * W[k*n+i,c] for k=0..p-1
                # So A is a p x rank matrix, and we want A @ U[i*m+j, :]^T = ones(p)
                
                U_new = U_curr.copy()
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                        # Solve A @ x = ones(p) in least squares
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = x
                        except:
                            pass
                U_curr = U_new
                
                # Solve for V: for each (j,k), sum_c V[j*p+k,c] * (U[i*m+j,c]*W[k*n+i,c]) = 1 for all i
                V_new = V_curr.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = x
                        except:
                            pass
                V_curr = V_new
                
                # Solve for W: for each (k,i), sum_c W[k*n+i,c] * (U[i*m+j,c]*V[j*p+k,c]) = 1 for all j
                W_new = W_curr.copy()
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = x
                        except:
                            pass
                W_curr = W_new
                
                # Check convergence
                if it % 20 == 19:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                                err = max(err, abs(val - 1.0))
                    if err < 1e-4:
                        converged = True
                        break
            
            if converged or (it >= 199):
                # Final check
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err = max(err, abs(val - 1.0))
                
                if err < 1e-3:
                    # Convert to list format
                    U_list = U_curr.tolist()
                    V_list = V_curr.tolist()
                    W_list = W_curr.tolist()
                    
                    # Try to round to half-integers or integers if close
                    U_rounded = []
                    V_rounded = []
                    W_rounded = []
                    can_round = True
                    for row in U_list:
                        new_row = []
                        for val in row:
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) < 1e-2:
                                new_row.append(nearest)
                            else:
                                can_round = False
                                new_row.append(val)
                        U_rounded.append(new_row)
                    
                    if can_round:
                        for row in V_list:
                            new_row = []
                            for val in row:
                                nearest = round(val * 2) / 2
                                if abs(val - nearest) < 1e-2:
                                    new_row.append(nearest)
                                else:
                                    can_round = False
                                    new_row.append(val)
                            V_rounded.append(new_row)
                    
                    if can_round:
                        for row in W_list:
                            new_row = []
                            for val in row:
                                nearest = round(val * 2) / 2
                                if abs(val - nearest) < 1e-2:
                                    new_row.append(nearest)
                                else:
                                    can_round = False
                                    new_row.append(val)
                            W_rounded.append(new_row)
                    
                    if can_round:
                        candidate = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    else:
                        candidate = {"U": U_list, "V": V_list, "W": W_list}
                    
                    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                        best = candidate
                        best_rank = rank
                        break  # Found valid decomposition at this rank, move to next rank
        
        # If we found a valid decomposition at this rank, we can stop (we go from low to high)
        if best_rank == rank:
            break
    
    return best
# EVOLVE-BLOCK-END
