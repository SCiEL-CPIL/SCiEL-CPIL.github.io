"""Matrix multiplication 2x3 by 3x4 - rank minimization via structured search."""
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Build the tensor T where T[(i,j),(j,k),(k,i)] = 1
    # For decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[(i,j),(j,k),(k,i)]
    # We need: sum_c u[i*m+j,c] * v[j*p+k,c] * w[k*n+i,c] = 1 if (i,j,k) is valid, else 0
    
    # The tensor has 24 entries (one for each i in [0,1], j in [0,1,2], k in [0,1,2,3])
    # Each entry must equal 1.
    
    # Schoolbook gives rank 24.
    # Let's try to find a lower rank decomposition.
    
    # Key insight: The constraint is that for each (i,j,k):
    #   sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c] = 1
    # And for entries not corresponding to valid (i,j,k) tuples, the sum must be 0.
    
    # Actually, in matrix multiplication C = A*B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has entries only for (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    # There are n*m*p = 24 such entries, each = 1.
    
    # The flattening approach: For each (i,k) pair, we need:
    #   sum_j A[i,j] B[j,k] = C[i,k]
    # In terms of our decomposition:
    #   sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c] summed over j = 1
    # Wait, that's not right. Let me re-read the task.
    
    # "U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]" summed over c should equal the tensor value.
    # The tensor value is 1 for each valid (i,j,k).
    
    # So for each triple (i,j,k): sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c] = 1
    
    # This is a system of 24 equations in rank variables (3*rank unknowns if we consider all entries).
    # But we can search over small ranks.
    
    # Known results: 
    # - 2x3 by 3x4 matrix multiplication
    # - The rank is bounded by n*m*p = 24
    # - Lower bounds: max(n*p, m*p, n*m) = max(8, 12, 6) = 12
    # - Upper bound via various algorithms
    
    # Let's try rank values from 12 upward.
    
    deadline = time.time() + 110  # Leave 10s margin
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def to_half_integers(U, V, W, tol=1e-6):
        """Try to snap values to half-integers."""
        U2 = [[round(x*2)/2 for x in row] for row in U]
        V2 = [[round(x*2)/2 for x in row] for row in V]
        W2 = [[round(x*2)/2 for x in row] for row in W]
        # Check if snapping is close enough
        for c in range(len(U)):
            for r in range(len(U[0])):
                if abs(U[c][r] - U2[c][r]) > tol:
                    return None
                if abs(V[c][r] - V2[c][r]) > tol:
                    return None
                if abs(W[c][r] - W2[c][r]) > tol:
                    return None
        return (U2, V2, W2)
    
    # Try known small-rank decompositions by construction
    # Strategy: For rank r, we need to assign integer/half-integer values to
    # U (n*m x r), V (m*p x r), W (p*n x r) such that all 24 constraints hold.
    
    # Let's use a systematic approach for small ranks.
    # For each column c, we have vectors u_c (length n*m), v_c (length m*p), w_c (length p*n).
    # Constraint: for each (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a bilinear (actually trilinear) system.
    
    # Let's try a different approach: fix U and V, solve for W, or use algebraic methods.
    
    # For the 2x3 x 3x4 case, let's try specific structured decompositions.
    
    # Approach: Try rank from lower bound up, using random search with half-integer values.
    
    best_result = None
    best_rank = n * m * p  # 24
    
    # First, get the schoolbook decomposition as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb
    
    # Try to improve with random search
    # Values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for rank in range(12, best_rank + 1):
        if time.time() > deadline:
            break
        
        # For each rank, try random assignments
        n_trials = 5000 if rank <= 15 else 2000
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            U_try = [[random.choice(val_set) for _ in range(rank)] for _ in range(n*m)]
            V_try = [[random.choice(val_set) for _ in range(rank)] for _ in range(m*p)]
            W_try = [[random.choice(val_set) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U_try, V_try, W_try, rank):
                snapped = to_half_integers(U_try, V_try, W_try)
                if snapped is not None:
                    U_try, V_try, W_try = snapped
                    if verify(U_try, V_try, W_try, rank):
                        best_result = (U_try, V_try, W_try)
                        best_rank = rank
                        break
        
        if best_rank == rank:
            break
    
    if best_result is None:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    U_out, V_out, W_out = best_result
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
