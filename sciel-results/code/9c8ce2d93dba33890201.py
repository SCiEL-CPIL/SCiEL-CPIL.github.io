"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Verify that the decomposition correctly represents matrix multiplication."""
        # The tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # should equal 1 if (a,b,c) corresponds to (i,j,k) where
        # a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.
        nm, mp, pn = n * m, m * p, p * n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Expected value is 1
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(U.shape[1]))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some non-support entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = (k + 1) % p * n + i  # different k
                    if (k + 1) % p != k:  # avoid checking when p=1
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(U.shape[1]))
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Try to find a better decomposition using Laderman-inspired approach
    # For 4x4x9, we can split into three 4x4x3 blocks
    
    def laderman_4x4x3():
        """
        Decomposition for 4x4 * 4x3 = 4x3 matrix multiplication.
        Returns U, V, W matrices for this sub-problem.
        Uses schoolbook for now: rank = 4*4*3 = 48
        """
        nn, mm, pp = 4, 4, 3
        nmm, mpp, pnn = nn * mm, mm * pp, pp * nn
        rank = nmm * mpp * pnn // (mm * pp * nn) * mm * pp * nn  # just mm*pp*nn = 48
        rank = mm * pp * nn  # 4*3*4 = 48
        
        U = np.zeros((nmm, rank))
        V = np.zeros((mpp, rank))
        W = np.zeros((pnn, rank))
        
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    idx = i * mm * pp + j * pp + k
                    a = i * mm + j
                    b = j * pp + k
                    c = k * nn + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
        return U, V, W, rank
    
    # Try combining three 4x4x3 decompositions
    def combined_decomposition():
        """
        Combine three 4x4x3 decompositions for the full 4x4x9 problem.
        Total rank = 3 * 48 = 144 (same as schoolbook for now)
        """
        U1, V1, W1, r1 = laderman_4x4x3()
        U2, V2, W2, r2 = laderman_4x4x3()
        U3, V3, W3, r3 = laderman_4x4x3()
        
        total_rank = r1 + r2 + r3
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Block 1: columns 0-2 of B
        U[:, 0:r1] = U1
        V[:, 0:r1] = V1
        W[:, 0:r1] = W1
        
        # Block 2: columns 3-5 of B
        # Need to shift indices
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    old_idx = i * 4 * 3 + j * 3 + k
                    new_idx = r1 + old_idx
                    a = i * 4 + j
                    b = j * 9 + (k + 3)  # columns 3,4,5
                    c = k * 4 + i
                    U[a, new_idx] = U1[i * 4 + j, old_idx]
                    V[b, new_idx] = V1[j * 3 + k, old_idx]
                    W[c, new_idx] = W1[k * 4 + i, old_idx]
        
        # Block 3: columns 6-8 of B
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    old_idx = i * 4 * 3 + j * 3 + k
                    new_idx = r1 + r2 + old_idx
                    a = i * 4 + j
                    b = j * 9 + (k + 6)  # columns 6,7,8
                    c = k * 4 + i
                    U[a, new_idx] = U1[i * 4 + j, old_idx]
                    V[b, new_idx] = V1[j * 3 + k, old_idx]
                    W[c, new_idx] = W1[k * 4 + i, old_idx]
        
        return U, V, W, total_rank
    
    # Use schoolbook as fallback
    U, V, W, rank = schoolbook()
    
    # Verify the schoolbook decomposition
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
