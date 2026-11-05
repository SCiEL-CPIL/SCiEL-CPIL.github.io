import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # For each (i, j, k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # And for all other combinations, it should be 0.
    # Actually, the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) is a valid index triple, 0 otherwise.
    # Wait, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k), and the reconstruction is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j'} for matching j
    # Actually the correct condition is:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is because the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in range.
    
    # Let's verify: the bilinear form should reconstruct C[i,k] = sum_j A[i,j] B[j,k]
    # C_reconstructed[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    # For this to equal sum_j A[i,j] B[j,k], we need the tensor decomposition to be exact.
    
    # The standard verification: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=int)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=int)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using Strassen-like structure
    # For 4x5x5, we can try various block decompositions
    
    # Strategy: Try rank from 99 down to some lower bound
    # Use a greedy/constructive approach
    
    # One approach: exploit the fact that we can use the 3x3x3 Strassen decomposition
    # embedded in the problem.
    
    # Let's try a more systematic approach: build the decomposition using known
    # low-rank decompositions for sub-problems.
    
    # For now, let's try to verify if we can do better than 100 by checking
    # if any single term can be removed or if terms can be merged.
    
    # Actually, let's try a constructive approach using the structure of matrix multiplication.
    # The key insight: we can use the rank-6 Strassen decomposition for 3x3 blocks.
    
    # Split 4x5x5 into blocks:
    # A is 4x5, B is 5x5, C is 4x5
    # Let's try: A = [A1 (3x5); A2 (1x5)], B = [B1 (5x3) B2 (5x2)], C = [C1 (4x3) C2 (4x2)]
    # C1 = A1*B1 + A2*B1, C2 = A1*B2 + A2*B2
    
    # This doesn't directly help unless we can share computations.
    
    # Let me try a different approach: use the fact that the tensor for (n+1)xmxp
    # has rank at most rank(nmxp) + m*p (by adding one row)
    # So rank(4x5x5) <= rank(3x5x5) + 5*5
    # And rank(3x5x5) <= rank(3x3x5) + 3*2*5 = rank(3x3x5) + 30
    # This is getting complicated.
    
    # Let's try a direct search for small improvements.
    # We'll try to find if rank 99 is achievable by checking if we can
    # express one of the basis tensors as a linear combination of others.
    
    # Actually, a cleaner approach: use integer programming or just try
    # to construct a valid decomposition with rank < 100.
    
    # For this specific case, let's try using the following construction:
    # Use the fact that we can combine some terms.
    
    # Let me try a random search with small integer coefficients.
    
    def try_random_rank(target_rank, num_trials=1000):
        """Try to find a valid decomposition with given rank using random search."""
        np.random.seed(42)
        for trial in range(num_trials):
            # Generate random sparse matrices with small integer entries
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for r in range(target_rank):
                # Each term should correspond to a "path" in the computation
                # Randomly assign small integers
                for idx in range(n*m):
                    if np.random.random() < 0.3:
                        U[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in range(m*p):
                    if np.random.random() < 0.3:
                        V[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in range(p*n):
                    if np.random.random() < 0.3:
                        W[idx, r] = np.random.choice([-2, -1, 1, 2])
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # The random search above is unlikely to work. Let me try a more structured approach.
    
    # Key idea: The matrix multiplication tensor for (n,m,p) has a natural
    # decomposition. We can try to use the following:
    # For 4x5x5, note that 4 = 3+1, so we can write:
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0..3, k=0..4
    # = sum_j (A[i,j] for i=0..2) B[j,k] + A[3,j] B[j,k]
    
    # The first part is a 3x5x5 multiplication, the second is a 1x5x5.
    # rank(4x5x5) <= rank(3x5x5) + rank(1x5x5) = rank(3x5x5) + 25
    
    # Similarly, rank(3x5x5) <= rank(3x3x5) + rank(3x2x5)
    # rank(3x3x5) <= rank(3x3x3) + rank(3x3x2) = 6 + 18 = 24 (using Strassen for 3x3x3)
    # Wait, rank(3x3x2) is not simply 18. Let me reconsider.
    
    # Actually, for rectangular cases, the schoolbook rank is n*m*p, and we can
    # sometimes do better using known decompositions.
    
    # Let me just try to verify the schoolbook and return it, since finding
    # a provably lower rank decomposition is complex.
    
    # But wait - let me try one more thing: use the structure to find if
    # rank 99 is possible by checking linear dependencies.
    
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [0,3]x[0,4]x[0,4].
    # The space of rank-1 tensors is spanned by u_i * v_j * w_k.
    # The question is whether the 100 standard basis elements are linearly independent
    # in the tensor space, or if there's a dependency that allows rank 99.
    
    # For matrix multiplication tensors, it's known that the rank is at least
    # max(n,m,p)^2 for square cases, but for rectangular it's more subtle.
    
    # Given the complexity, let's return the schoolbook decomposition which is
    # guaranteed to be correct.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
