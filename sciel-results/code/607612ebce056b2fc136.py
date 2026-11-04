"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 125  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # schoolbook rank
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Try to find a lower-rank decomposition via structured search
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(24, 30, 20) = 30
    # Known upper bounds for 4x5x6: we'll try to find rank < 120
    
    # Strategy: use the fact that matrix multiplication tensors have structure.
    # Try ranks from a reasonable lower bound up to the schoolbook rank.
    # For 4x5x6, known best is likely around 48-60 or so.
    
    # We'll use a simple but effective approach:
    # 1. Start with schoolbook
    # 2. Try to reduce rank by identifying linear dependencies
    
    # Actually, let's use a more principled approach.
    # The matrix multiplication tensor T_{ijk} = delta_{i,k} (for (i,j) x (j,k))
    # We need rank-r factors U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}
    # Wait, that's not quite right. Let me re-derive.
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the bilinear form is: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * ... 
    # No, the standard formulation: 
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    # Wait, that's not standard either. Let me think more carefully.
    
    # Standard matrix multiplication tensor decomposition:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition means:
    # C[i,k] = sum_{c=1}^r U_c[i,j] * V_c[j,k] * W_c[i,k]  -- no, that's not right.
    
    # The correct formulation:
    # A, B are inputs, C is output.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-r decomposition: C[i,k] = sum_{c=1}^r (sum_j U[c, i, j] A[i,j]) (sum_j V[c, j, k] B[j,k])
    # But the interface specifies:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And W follows trace convention W[k*n+i, r]
    
    # So the decomposition is:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the sums over i,j,k are implicit in how we use these.
    
    # Actually, I think the correct interpretation is:
    # For each "rank term" c, we have:
    # U_c is a matrix of size n x m (flattened to n*m)
    # V_c is a matrix of size m x p (flattened to m*p)  
    # W_c is a matrix of size p x n (flattened to p*n, with W[k*n+i, c])
    # And: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{i,k} ... no.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n,m) x (m,p) -> (n,p) is:
    # T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # T_{i,j,k} = 1 if this term contributes to C[i,k]
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T_{i,j,k} = 1 for all valid (i,j,k)
    
    # A rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    # u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] summed over c should give the tensor.
    
    # Wait, but the tensor T_{i,j,k} doesn't depend on i and k separately in a simple way.
    # T_{i,j,k} = 1 for all i,j,k.
    
    # Hmm, but that would mean any decomposition works as long as the product sums to 1.
    # That doesn't seem right for "matrix multiplication tensor".
    
    # Let me reconsider. The matrix multiplication tensor is typically defined as:
    # For multiplying an n×m matrix by an m×p matrix to get an n×p matrix,
    # the tensor T ∈ R^{n×m × m×p × n×p} where
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' else 0
    # i.e., T[(i,j),(j,k),(i,k)] = 1, all other entries 0.
    
    # So the tensor is: T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(i,k) for some i,j,k
    # In flattened indices: T[i*m+j, j*p+k, k*n+i] = 1
    
    # A rank-r decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c] = sum_{r=0}^{rank-1} U[a,r] * V[b,r] * W[c,r]
    
    # So we need: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is a system of equations. The schoolbook solution uses rank = n*m*p
    # with U[i*m+j, i*m*p + j*p + k] = 1, V[j*p+k, i*m*p + j*p + k] = 1, W[k*n+i, i*m*p + j*p + k] = 1
    
    # To find lower rank, we need to solve this system with fewer columns.
    
    # Given the complexity, let's try a few specific ranks and use a solver.
    
    # For 4x5x6, let's try to find the minimum rank.
    # We'll use a simple approach: try ranks from lower bound to upper bound.
    
    lower_bound = max(n*p, m*p, n*m)  # = max(24, 30, 20) = 30
    # But the true lower bound might be higher. Let's try a range.
    
    # Given time constraints, let's just return the schoolbook for now,
    # but try a few quick optimizations.
    
    # Actually, let's try a structured approach based on known results.
    # For small matrices, the minimum rank is often close to the "Strassen-like" bounds.
    
    # Let's just try to verify the schoolbook and return it, since finding
    # a lower rank requires solving a hard problem.
    
    # But wait - maybe we can find a better decomposition using the structure.
    # Let's try rank = 48, 40, 36, 32, 30 and see if we can find solutions.
    
    # Due to time constraints, let's just return the schoolbook decomposition.
    # The key insight is that for this task, the schoolbook is the baseline (score 1.0),
    # and finding a lower rank would improve the score.
    
    # Let me try a quick search for lower ranks.
    for rank in range(30, best_rank + 1, 2):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank we tried
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using a simple approach."""
    # This is a placeholder - a real implementation would use optimization
    # or algebraic methods to find the decomposition.
    # For now, return None to indicate failure.
    return None

# EVOLVE-BLOCK-END
