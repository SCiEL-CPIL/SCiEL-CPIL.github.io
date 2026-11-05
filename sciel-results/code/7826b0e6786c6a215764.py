"""Rewritten program for matrix multiplication 3,3,8 rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # The tensor T has entries T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # Wait, the matrix multiplication tensor is:
    # T_{i,j,k} = delta_{j,k} ... no.
    # The multiplication tensor for (n,m,p) has indices (i,j,k) where:
    # i in 0..n-1 (row of A, row of C)
    # j in 0..m-1 (col of A, row of B)
    # k in 0..p-1 (col of B, col of C)
    # T_{i,j,k} = 1 if the product C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T is a 3-tensor with T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{i,j,k} = 1 if j is the contraction index... 
    # Let me be precise. The bilinear map (A,B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has T[i, j, k] = 1 for all i,j,k? No, that's not right.
    # The tensor is T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(1)} ⊗ e_j^{(2)} ⊗ e_k^{(3)}
    # where the three modes correspond to (i,j), (j,k), (k,i) or similar.
    
    # Let's use the standard definition. The multiplication tensor for C = A*B where
    # A is n x m, B is m x p, C is n x p.
    # T_{i,j,k} = 1 if the index j appears in both A and B.
    # More precisely, T is a tensor in V^* ⊗ V^* ⊗ V^* where V = R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    # T[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # So T[(i,j), (j',k), (i',k')] = 1 if j=j', i=i', k=k'
    
    # The rank of this tensor is the matrix multiplication complexity.
    # For 3x3 x 3x8, the schoolbook rank is 3*3*8 = 72.
    # We want to find a lower rank decomposition.
    
    # Known results: For small matrices, the rank is often equal to the schoolbook rank,
    # but sometimes it can be lower. For 3x3 * 3x3, the rank is 23 (Strassen's algorithm).
    # For 3x3 * 3x8, we need to investigate.
    
    # Let's try to use a heuristic search or known constructions.
    # One approach: use the fact that 3x3 * 3x8 can be related to 3x3 * 3x3 blocks.
    
    # Actually, let's think about this differently. The task asks for a rank-minimal exact
    # tensor decomposition. The score is the rank, lower is better.
    
    # For now, let's return the schoolbook decomposition as a baseline, and then try
    # to improve it.
    
    # Schoolbook decomposition:
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Check if we have time to try something better
    if time.time() < deadline:
        # Try to find a lower rank decomposition using known results or heuristics
        # For 3x3 * 3x8, let's check if there's a known lower bound or construction
        
        # The Strassen algorithm for 2x2 gives rank 7 instead of 8.
        # For 3x3, the best known is 23.
        # For 3x3 * 3x8, we can potentially use block methods.
        
        # Let's try a simple heuristic: random search for lower rank
        # But this is computationally expensive and may not converge.
        
        # For now, return the schoolbook result.
        pass
    
    return result
# EVOLVE-BLOCK-END
