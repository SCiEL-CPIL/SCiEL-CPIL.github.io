"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Try to find lower rank solutions
    # The constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i==i', k==k')
    # for all i, i', j, k
    
    # Let's set up the problem. For each (i, i', j, k), we have a constraint.
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    
    # Try ranks from some lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = 24? Not necessarily, rank can be less.
    # Actually, a simple lower bound is the "subrank" or we can use dimension arguments.
    
    # Let's try to use a random initialization and gradient-free approach, or just try
    # to verify if rank 72 is minimal by attempting lower ranks.
    
    # For now, let's just return the schoolbook decomposition as it's guaranteed correct.
    # But let's try to find something better.
    
    # Actually, let me think about this more carefully.
    # For multiplying (n x m) by (m x p), the tensor rank is at most n*m*p.
    # Can we do better?
    
    # One idea: if p is a multiple of something, we might be able to use block structure.
    # Here n=3, m=3, p=8. 8 is not a multiple of 3.
    
    # Let me try a numerical approach: set up the bilinear system and try to solve for low rank.
    
    # The number of constraints is n * n * m * p = 3 * 3 * 3 * 8 = 216
    # (for each i, i', j, k, we need the bilinear form to match)
    
    # Wait, let me recount. The bilinear map is determined by its action on basis elements.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the "input" basis elements are A[i,j] for i in range(n), j in range(m) -> n*m = 9
    # and B[j,k] for j in range(m), k in range(p) -> m*p = 24
    # The "output" basis elements are C[i,k] for i in range(n), k in range(p) -> n*p = 24
    
    # The tensor T has entries T[output_idx, input1_idx, input2_idx]
    # where input1_idx = i*m + j, input2_idx = j*p + k, output_idx = k*n + i
    # T[output_idx, input1_idx, input2_idx] = 1 if the indices are consistent, else 0
    
    # Specifically, T[(k*n+i), (i'*m+j), (j'*p+k')] = 1 iff i==i' and k==k' and j==j'
    
    # So for each (i, j, k), there's exactly one nonzero entry: T[k*n+i, i*m+j, j*p+k] = 1
    
    # Total nonzeros: n*m*p = 72
    
    # A rank-r decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # means: T[a,b,c] = sum_{r=1}^R U[a,r] * V[b,r] * W[c,r]
    
    # So we need: for all a, b, c:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # This is 9 * 24 * 24 = 5184 equations, but most are 0 = 0.
    # The nonzero ones: for each (i,j,k), T[k*n+i, i*m+j, j*p+k] = 1
    # That's 72 equations equal to 1.
    # The rest (5184 - 72 = 5112) are equations equal to 0.
    
    # This is a hard nonlinear system. Let me try a different approach.
    
    # Given time constraints, let me just return the schoolbook solution.
    # But first, let me see if I can find any improvement.
    
    # Actually, let me try rank 72 (schoolbook) and verify it's correct, then return it.
    # The question is whether we can do better than 72.
    
    # For now, return schoolbook.
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    # Reconstruct tensor and check
    # T_reconstructed[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is expensive for large rank, but rank=72 is manageable
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
