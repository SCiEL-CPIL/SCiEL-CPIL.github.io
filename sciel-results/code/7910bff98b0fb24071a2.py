"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        # Check all entries of the tensor
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k', c = k''*n+i'
        # T should be 1 iff i==i', j==j', k==k''
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                    val = np.dot(U[a], V[b] * W[c])
                    if val != expected:
                        return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def winograd_1x4x4():
        """
        Winograd scheme for 1x4x4 convolution-like problem.
        This gives a rank-4 decomposition for computing 4 outputs from 4 inputs
        with a 4-tap filter, using only 4 multiplications instead of 16.
        
        The idea: for computing y[i] = sum_j A[j] * B[i+j] for i=0..3, j=0..3,
        we can use the transform:
          u = A[0] + A[1]
          v = A[1] + A[2]  
          w = A[2] + A[3]
          z = A[0] + A[3]
        and similar for B, then combine.
        
        Actually, let me use a known rank-4 scheme for 4-point convolution.
        """
        # For 1x4x4: compute C[i,k] = sum_j A[j] * B[i+j,k] 
        # But we need a specific structure. Let me think differently.
        
        # Simpler: use the fact that for small dimensions, we can find
        # low-rank decompositions by exploiting linear dependencies.
        
        # For a 2x2x2 block, rank is 2 (not 8) using:
        # C[0,0] = A[0]*B[0]
        # C[0,1] = A[0]*B[1]
        # C[1,0] = A[1]*B[0]  
        # C[1,1] = A[1]*B[1]
        # This is already rank 4 for 2x2x2.
        
        # For 3x3x3, the minimal rank is known to be 23 (over reals), 
        # but we need integer/half-integer.
        
        return None  # Placeholder

    def try_winograd_block(n_block, m_block, p_block):
        """
        Try to find a low-rank decomposition for an n_block x m_block x p_block
        subproblem using Winograd-like techniques.
        
        Returns (U, V, W, rank) or None if not found.
        """
        # For now, use schoolbook for blocks
        r = n_block * m_block * p_block
        U = np.zeros((n_block * m_block, r), dtype=np.int64)
        V = np.zeros((m_block * p_block, r), dtype=np.int64)
        W = np.zeros((p_block * n_block, r), dtype=np.int64)
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i * m_block + j, col] = 1
                    V[j * p_block + k, col] = 1
                    W[k * n_block + i, col] = 1
                    col += 1
        return U, V, W, r

    def combine_decompositions(decomp_list, n, m, p):
        """
        Combine multiple block decompositions into a full decomposition.
        decomp_list: list of (U, V, W, rank, i_offset, j_offset, k_offset)
        """
        total_rank = sum(d[3] for d in decomp_list)
        U = np.zeros((n * m, total_rank), dtype=np.int64)
        V = np.zeros((m * p, total_rank), dtype=np.int64)
        W = np.zeros((p * n, total_rank), dtype=np.int64)
        
        col_offset = 0
        for (Ub, Vb, Wb, rank, i_off, j_off, k_off) in decomp_list:
            # Place block U at rows corresponding to (i_off..i_off+nb-1, j_off..j_off+mb-1)
            nb, mb = Ub.shape[0] // mb if mb > 0 else 1, 1
            # Actually, need to know block sizes
            pass  # This needs more careful implementation
        
        return U, V, W, total_rank

    # Start with schoolbook
    U_sb, V_sb, W_sb, base_rank = schoolbook()
    current_best = (base_rank, U_sb, V_sb, W_sb)
    
    # Try to improve using structured search
    # Strategy: look for patterns that allow rank reduction
    
    # For 3x5x7, let's try a direct approach:
    # The tensor has 3*5*7 = 105 ones.
    # We need to find U, V, W such that the trilinear form matches.
    
    # Try random search with structured initialization
    def structured_random_search(max_rank, num_trials=100):
        """Random search with better initialization."""
        best = None
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            r = max_rank
            # Initialize with schoolbook-like structure but randomize
            U = np.zeros((n * m, r), dtype=np.int64)
            V = np.zeros((m * p, r), dtype=np.int64)
            W = np.zeros((p * n, r), dtype=np.int64)
            
            # Assign random small integers
            for i in range(n * m):
                for j in range(r):
                    U[i, j] = np.random.randint(-2, 3)
            for i in range(m * p):
                for j in range(r):
                    V[i, j] = np.random.randint(-2, 3)
            for i in range(p * n):
                for j in range(r):
                    W[i, j] = np.random.randint(-2, 3)
            
            if verify(U, V, W):
                if best is None or r < best[0]:
                    best = (r, U, V, W)
        
        return best

    # Try a few ranks below schoolbook
    for r in range(base_rank - 1, max(base_rank - 20, 80), -1):
        if time.time() > deadline - 30:
            break
        found = structured_random_search(r, num_trials=20)
        if found is not None:
            if r < current_best[0]:
                current_best = found
                break
    
    rank, U, V, W = current_best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
