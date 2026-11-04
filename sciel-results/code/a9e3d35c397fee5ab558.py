import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T
    # T[u, v, w] where u in [0, nm), v in [0, mp), w in [0, pn)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u else 0
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True

    def verify_full(U, V, W, rank):
        """Full tensor verification."""
        # Flatten U, V, W
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Compute sum over c: U[u,c] V[v,c] W[w,c]
        # This is not a simple matrix mult due to index mismatch in standard batched matmul
        # We can use einsum
        # result[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a 3-way tensor product sum
        # Can be done as: 
        # Temp[u,v,c] = U[u,c] * V[v,c]
        # result[u,v,w] = sum_c Temp[u,v,c] * W[w,c]
        
        # Efficient way:
        # Loop over w
        for w in range(pn):
            # W[w, :] is a vector of length rank
            # We need sum_c (U[:, c] * V[:, c]) * W[w, c]
            # Let M[u,v] = sum_c U[u,c] V[v,c] W[w,c]
            # This is equivalent to:
            # For each c, add U[:, c] outer V[:, c] scaled by W[w, c]
            M = np.zeros((nm, mp), dtype=np.int64)
            for c in range(rank):
                if W_np[w, c] != 0:
                    M += U_np[:, c] * V_np[:, c] * W_np[w, c]
            if not np.array_equal(M, T[:, :, w]):
                return False
        return True

    # Search for lower rank
    # We use the linear solve for W given U, V
    # For each w, solve A_w W_w = b_w
    # A_w is nm*mp x rank
    # b_w is nm*mp
    
    # Precompute indices for T
    # T is sparse. Only nm*mp*p entries are 1? No, only nm*mp entries per w?
    # T[u,v,w] is 1 only if u,v,w are consistent.
    # For a fixed w = k*n + i, 
    # u = i*m + j, v = j*p + k.
    # So for fixed w, there are m non-zero entries in T[:, :, w].
    
    min_rank = 30 # Try down to 30
    # Use small integers for U, V
    # To improve chances, we can fix the first few columns of U, V to be standard basis?
    # Or just random.
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random candidates for this rank
        # Number of attempts depends on time
        max_attempts = 100
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V with small integers
            # Using -1, 0, 1 increases chance of linear independence and sparsity
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
            
            U_np = np.array(U, dtype=np.int64)
            V_np = np.array(V, dtype=np.int64)
            
            W = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for each w
            for w in range(pn):
                # Construct A_w (nm*mp x target_rank)
                # A_w[u*mp + v, c] = U[u, c] * V[v, c]
                # This is large. nm*mp = 6*21 = 126.
                # We can solve the linear system A_w W_w = b_w
                # b_w is the slice T[:, :, w]
                
                # Instead of building full 126 x r matrix, we can use least squares or check consistency
                # Since we want exact integer solutions, we can try to solve
                # But 126 equations for r=41 unknowns is overdetermined.
                # We can use numpy.linalg.lstsq to find a solution, then check if it's integer.
                
                A_w = np.zeros((nm * mp, target_rank), dtype=np.float64)
                for u in range(nm):
                    for v in range(mp):
                        A_w[u * mp + v, :] = U_np[u, :] * V_np[v, :]
                
                b_w = T[:, :, w].flatten().astype(np.float64)
                
                # Solve A_w W_w = b_w
                # Use least squares
                try:
                    W_w, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integer/half-integer
                # And check residual
                if residuals.size > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
                    
                # Snap to nearest half-integer?
                # If the solution is rational with small denominator, snapping might work.
                # For now, check if it's close to integer.
                W_w_snapped = np.round(W_w * 2) / 2.0
                
                # Verify snapped solution
                residual_check = np.linalg.norm(A_w.dot(W_w_snapped) - b_w)
                if residual_check > 1e-6:
                    valid = False
                    break
                
                # Store W_w
                for c in range(target_rank):
                    W[w][c] = W_w_snapped[c]
            
            if valid:
                # Final verification with full check
                if verify_full(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    return best
# EVOLVE-BLOCK-END
