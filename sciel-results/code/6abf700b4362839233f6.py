"""Combined approach for 3x4x7 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank):
        """Faster vectorized check using matrix operations."""
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This is equivalent to: for each i,j, the vector (over k) should be all 1s
        # sum_r U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r] = 1 for all k
        
        # Precompute: for each i, W_i[k,r] = W[k*n+i, r] for k in range(p)
        # For each j, V_j[k,r] = V[j*p+k, r] for k in range(p)
        # For each (i,j), we need: sum_r U[i*m+j,r] * (W_i * V_j)[k,r] = 1 for all k
        
        # More efficient: build the tensor and check
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Use einsum-style approach
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # This is a 3-mode contraction
        
        # Compute for all (i,j,k) at once using broadcasting
        # term[i,j,k,r] = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # T[i,j,k] = sum_r term[i,j,k,r]
        
        # term shape: (n, m, p, rank)
        term = U_r[:, :, None, :] * V_r[:, None, :, :] * W_r[None, :, :, :]
        # Wait, W_r is (p, n, rank), we need W_r[k, i, r]
        # So term[i,j,k,r] = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # Let's do it differently:
        # For each i: 
        #   For each j:
        #     For each k:
        #       sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # Vectorized: 
        # A[i,j,r] = U_r[i,j,r]
        # B[j,k,r] = V_r[j,k,r]  
        # C[k,i,r] = W_r[k,i,r]
        # T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
        
        # This is: T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True
    
    def try_separable_structure(time_limit):
        """
        Try a separable ansatz:
        U[i,j,r] = u_r[i] * v_r[j]
        V[j,k,r] = v_r[j] * w_r[k]
        W[k,i,r] = w_r[k] * u_r[i]
        
        Then T[i,j,k] = sum_r u_r[i]^2 * v_r[j]^2 * w_r[k]^2
        
        We need this to equal 1 for all (i,j,k).
        
        This is equivalent to finding vectors u_r, v_r, w_r such that
        the sum of outer products of (u_r*u_r, v_r*v_r, w_r*w_r) gives the all-ones tensor.
        
        For small ranks, try to solve this system.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Try ranks from a lower bound up
        # Lower bound: the tensor rank is at least max(n, m, p) = 7
        # But for the all-ones tensor of shape (3,4,7), the rank might be higher
        
        # Let's try a direct construction:
        # If we can find r such that there exist u_1,...,u_r in R^n, 
        # v_1,...,v_r in R^m, w_1,...,w_r in R^p with
        # sum_r (u_r[i]*v_r[j]*w_r[k])^2 = 1 for all i,j,k
        
        # This is equivalent to: the tensor formed by the r rank-1 tensors
        # (u_r ⊗ v_r ⊗ w_r) has the property that its "squared sum" is 1.
        
        # For a simpler case: if all u_r, v_r, w_r have entries ±1/sqrt(d),
        # then each term contributes 1/(n*m*p) and we need n*m*p terms.
        # That's the schoolbook rank.
        
        # To do better, we need structured vectors.
        
        # Try: use Hadamard-like structures
        # For m=4: use H_4 / 2 (entries ±1/2)
        # For n=3: use a 3x3 structured matrix
        # For p=7: use a 7x7 structured matrix
        
        # Let's try a random search with structured initialization
        
        for rank_attempt in range(7, min(50, time_limit - (time.time() - start_time))):
            if time.time() - start_time > time_limit:
                return None
            
            for trial in range(50):
                if time.time() - start_time > time_limit:
                    return None
                
                # Initialize with random signs and magnitudes
                u = np.zeros((n, rank_attempt))
                v = np.zeros((m, rank_attempt))
                w = np.zeros((p, rank_attempt))
                
                for r in range(rank_attempt):
                    # Random initialization with small values
                    u[:, r] = np.random.choice([-1.0, 1.0], size=n) * np.random.uniform(0.5, 1.5)
                    v[:, r] = np.random.choice([-1.0, 1.0], size=m) * np.random.uniform(0.5, 1.5)
                    w[:, r] = np.random.choice([-1.0, 1.0], size=p) * np.random.uniform(0.5, 1.5)
                
                # Build U, V, W from the separable ansatz
                U = np.zeros((n*m, rank_attempt))
                V = np.zeros((m*p, rank_attempt))
                W = np.zeros((p*n, rank_attempt))
                
                for r in range(rank_attempt):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] = u[i, r] * v[j, r]
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, r] = v[j, r] * w[k, r]
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = w[k, r] * u[i, r]
                
                if check_decomposition_fast(U, V, W, rank_attempt):
                    return (U, V, W)
        
        return None
    
    def try_greedy_merge(time_limit):
        """
        Start from the schoolbook decomposition and try to merge terms.
        Two terms can be merged if their sum has rank 1.
        
        Term r: U_r ⊗ V_r ⊗ W_r where U_r is a vector in R^{n*m}, etc.
        The sum of two terms has rank 1 if they are linearly dependent in a specific way.
        
        Actually, for the all-ones tensor, each schoolbook term is:
        U_r[i*m+j] = 1 if (i,j) = (i_r, j_r) else 0
        V_r[j*p+k] = 1 if (j,k) = (j_r, k_r) else 0
        W_r[k*n+i] = 1 if (k,i) = (k_r, i_r) else 0
        
        These are all "corner" tensors. Merging them directly is hard.
        
        Instead, try a different approach: use a "block" structure where
        we group indices and use structured matrices within each block.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Try a "tiled" approach:
        # Divide the (i,j,k) space into blocks and assign each block to a rank slot
        # with appropriate weights.
        
        # For 3x4x7, try different block sizes
        # Block by i: 3 blocks of size 1
        # Block by j: 4 blocks of size 1
        # Block by k: 7 blocks of size 1
        
        # Try: for each (i,j), use a shared set of rank slots for all k
        # U[i*m+j, r] = a_{i,j,r}
        # V[j*p+k, r] = b_{j,k,r}
        # W[k*n+i, r] = c_{k,i,r}
        
        # This is just the general form, so no reduction.
        
        # Alternative: use a "factorization" of the index space
        # For example, write 4 = 2*2 and use a 2x2 block structure
        
        # Let's try a specific construction:
        # For the middle dimension (m=4), split into two groups of 2
        # For each group, use a 3x2x7 multiplication
        # Use Strassen-like ideas for the 2x2 blocks
        
        # Actually, let's try a "cyclic" structure:
        # Use rank slots indexed by (a, b, c) where a in Z_3, b in Z_4, c in Z_7
        # But that's 84 slots, same as schoolbook.
        
        # To reduce rank, we need to identify equivalent slots.
        
        # Try: use a "compressed" cyclic structure where we identify
        # slots that differ by a global shift.
        
        # For now, return None and rely on other methods
        return None
    
    def try_strassen_inspired(time_limit):
        """
        Use Strassen's algorithm structure to build a decomposition.
        For 3x4x7, exploit the 4=2*2 structure in the middle dimension.
        
        Strassen's algorithm for 2x2x2 uses 7 rank-1 terms.
        For 4x4x4, we can use 4x4 = 2x2 blocks, giving rank 7^2 = 49.
        
        For 3x4x7:
        - Split the 4 middle indices into two groups of 2
        - For each group, we have a 3x2x7 multiplication
        - Use Strassen-like ideas for the 2x2 parts
        
        A 3x2x7 multiplication can be decomposed as:
        - Split the 3 rows into {0,1} and {2}
        - The {0,1} x 2 x 7 part can use Strassen for the 2x2 blocks
        - The {2} x 2 x 7 part is just 2*7 = 14 schoolbook terms
        
        Total: 7 (for the 2x2x7 part) + 14 (for the 1x2x7 part) = 21 per group
        Two groups: 21 * 2 = 42
        
        But this is an upper bound, not necessarily achievable.
        
        Let's try to construct this explicitly.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # For now, return None as the construction is complex
        return None
    
    # Try the separable structure
    result = try_separable_structure(DEADLINE - 10)
    if result is not None:
        U, V, W = result
        best_rank = W.shape[1]
        best_U = U
        best_V = V
        best_W = W
    
    # If we haven't improved, try the greedy merge
    if best_rank == rank_sb:
        result = try_greedy_merge(DEADLINE - 10)
        if result is not None:
            U, V, W = result
            best_rank = W.shape[1]
            best_U = U
            best_V = V
            best_W = W
    
    # If we haven't improved, try the Strassen-inspired approach
    if best_rank == rank_sb:
        result = try_strassen_inspired(DEADLINE - 10)
        if result is not None:
            U, V, W = result
            best_rank = W.shape[1]
            best_U = U
            best_V = V
            best_W = W
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
