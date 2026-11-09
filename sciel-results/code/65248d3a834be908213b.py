"""Program that finds a rank-minimal tensor decomposition for 2x3x8 matrix multiplication
using structured Strassen-like block decomposition."""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Return a valid tensor decomposition for 2x3x8 matrix multiplication.
    Uses Strassen on 2x2x2 blocks to reduce rank from 48 to 44.
    """
    start = time.time()
    deadline = start + 140  # safety margin

    n, m, p = 2, 3, 8

    # Split j into J1={0,1} and J2={2}
    # Split k into four pairs: K0={0,1}, K1={2,3}, K2={4,5}, K3={6,7}
    j_blocks = [[0, 1], [2]]
    k_blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]

    # Build Strassen 2x2x2 decomposition
    # For 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # Strassen's 7 multiplications:
    # M1 = A[0,0]*(B[1,0]+B[1,1])
    # M2 = (A[0,0]+A[0,1])*B[1,1]
    # M3 = (A[1,0]+A[1,1])*B[0,0]
    # M4 = A[1,1]*(B[0,0]-B[1,0])
    # M5 = (A[0,0]+A[1,1])*(B[0,1]+B[1,1])
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # M7 = (A[0,0]-A[0,1])*(B[0,0]+B[1,0])
    #
    # C[0,0] = M5 + M4 - M2 + M6
    # C[0,1] = M1 + M5
    # C[1,0] = M3 + M7
    # C[1,1] = M1 + M2 - M4 + M3
    #
    # For the tensor decomposition, we need U, V, W such that:
    # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    #
    # Each Strassen term r corresponds to a product of linear combinations.
    # For term M1: A[0,0]*(B[1,0]+B[1,1])
    #   U contribution: U[0*2+0, r] = 1 (for A[0,0])
    #   V contribution: V[1*2+0, r] = 1, V[1*2+1, r] = 1 (for B[1,0]+B[1,1])
    #   W contribution: W[0*2+0, r] = 1, W[1*2+0, r] = 1 (appears in C[0,0] and C[0,1])
    #
    # Actually, let me think more carefully. The tensor T[i,j,k] = 1 for all i,j,k.
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For the 2x2x2 subproblem with local indices (i', j', k') in {0,1}:
    # We need: sum_r U_local[i',j',r] * V_local[j',k',r] * W_local[k',i',r] = 1
    #
    # Strassen gives 7 terms. Let me define them explicitly.
    #
    # Term 0 (M1): A[0,0] * (B[1,0] + B[1,1])
    #   This contributes to C[0,0] and C[0,1]
    #   U: U[0,0]=1 (i'=0, j'=0)
    #   V: V[1,0]=1, V[1,1]=1 (j'=1, k'=0 and k'=1)
    #   W: W[0,0]=1, W[1,0]=1 (k'=0,i'=0 and k'=1,i'=0)
    #
    # But wait, the W factor depends on k and i. For M1 to contribute to both C[0,0] and C[0,1]:
    # When k'=0: W[0,0] should be 1
    # When k'=1: W[1,0] should be 1
    # So W[0*2+0, r] = 1 and W[1*2+0, r] = 1
    #
    # Let me just enumerate all 7 Strassen terms with their U, V, W coefficients.

    def strassen_2x2x2():
        """
        Return (U, V, W) for 2x2x2 tensor with rank 7.
        U shape: (2*2, 7) = (4, 7)
        V shape: (2*2, 7) = (4, 7)
        W shape: (2*2, 7) = (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: M1 = A[0,0]*(B[1,0]+B[1,1])
        # Contributes to C[0,0] and C[0,1]
        U[0*2+0, 0] = 1  # A[0,0]
        V[1*2+0, 0] = 1  # B[1,0]
        V[1*2+1, 0] = 1  # B[1,1]
        W[0*2+0, 0] = 1  # C[0,0] contribution
        W[1*2+0, 0] = 1  # C[0,1] contribution
        
        # Term 1: M2 = (A[0,0]+A[0,1])*B[1,1]
        # Contributes to C[0,0] and C[1,1]
        U[0*2+0, 1] = 1  # A[0,0]
        U[0*2+1, 1] = 1  # A[0,1]
        V[1*2+1, 1] = 1  # B[1,1]
        W[0*2+0, 1] = 1  # C[0,0] contribution
        W[1*2+1, 1] = 1  # C[1,1] contribution
        
        # Term 2: M3 = (A[1,0]+A[1,1])*B[0,0]
        # Contributes to C[1,0] and C[1,1]
        U[1*2+0, 2] = 1  # A[1,0]
        U[1*2+1, 2] = 1  # A[1,1]
        V[0*2+0, 2] = 1  # B[0,0]
        W[0*2+1, 2] = 1  # C[1,0] contribution
        W[1*2+1, 2] = 1  # C[1,1] contribution
        
        # Term 3: M4 = A[1,1]*(B[0,0]-B[1,0])
        # Contributes to C[0,0] and C[1,1]
        U[1*2+1, 3] = 1  # A[1,1]
        V[0*2+0, 3] = 1  # B[0,0]
        V[1*2+0, 3] = -1 # -B[1,0]
        W[0*2+0, 3] = 1  # C[0,0] contribution
        W[1*2+1, 3] = -1 # -C[1,1] contribution
        
        # Term 4: M5 = (A[0,0]+A[1,1])*(B[0,1]+B[1,1])
        # Contributes to C[0,0] and C[0,1]
        U[0*2+0, 4] = 1  # A[0,0]
        U[1*2+1, 4] = 1  # A[1,1]
        V[0*2+1, 4] = 1  # B[0,1]
        V[1*2+1, 4] = 1  # B[1,1]
        W[0*2+0, 4] = 1  # C[0,0] contribution
        W[1*2+0, 4] = 1  # C[0,1] contribution
        
        # Term 5: M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # Contributes to C[0,0] and C[0,1]
        U[0*2+1, 5] = 1  # A[0,1]
        U[1*2+1, 5] = -1 # -A[1,1]
        V[1*2+0, 5] = 1  # B[1,0]
        V[1*2+1, 5] = 1  # B[1,1]
        W[0*2+0, 5] = 1  # C[0,0] contribution
        W[1*2+0, 5] = 1  # C[0,1] contribution
        
        # Term 6: M7 = (A[0,0]-A[0,1])*(B[0,0]+B[1,0])
        # Contributes to C[1,0] and C[1,1]
        U[0*2+0, 6] = 1  # A[0,0]
        U[0*2+1, 6] = -1 # -A[0,1]
        V[0*2+0, 6] = 1  # B[0,0]
        V[1*2+0, 6] = 1  # B[1,0]
        W[0*2+1, 6] = 1  # C[1,0] contribution
        W[1*2+1, 6] = 1  # C[1,1] contribution
        
        return U, V, W

    # Verify Strassen decomposition
    def verify_2x2x2(U, V, W, tol=1e-12):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(7))
                    if abs(s - 1) > tol:
                        return False
        return True

    U_str, V_str, W_str = strassen_2x2x2()
    assert verify_2x2x2(U_str, V_str, W_str), "Strassen verification failed"

    # Now build the full 2x3x8 decomposition
    # Total rank = 4 * 7 (Strassen blocks) + 4 * 4 (schoolbook blocks) = 28 + 16 = 44
    total_rank = 4 * 7 + 4 * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)
    
    col = 0
    for kb_idx, k_block in enumerate(k_blocks):
        k0, k1 = k_block[0], k_block[1]
        
        # J1 = {0, 1} block: use Strassen (rank 7)
        for r in range(7):
            # Map local Strassen indices to global
            for li in range(2):  # local i
                gi = li  # global i
                for lj in range(2):  # local j in {0,1}
                    gj = lj  # global j
                    U[gi * m + gj, col] = U_str[li * 2 + lj, r]
            
            for lj in range(2):  # local j in {0,1}
                gj = lj  # global j
                for lk in range(2):  # local k
                    gk = [k0, k1][lk]  # global k
                    V[gj * p + gk, col] = V_str[lj * 2 + lk, r]
            
            for lk in range(2):  # local k
                gk = [k0, k1][lk]  # global k
                for li in range(2):  # local i
                    gi = li  # global i
                    W[gk * n + gi, col] = W_str[lk * 2 + li, r]
            
            col += 1
        
        # J2 = {2} block: use schoolbook (rank 4)
        # For i in {0,1}, j=2, k in {k0,k1}: T[i,2,k] = 1
        # One term per (i, k) pair
        for i in range(2):
            for lk in range(2):
                gk = [k0, k1][lk]
                U[i * m + 2, col] = 1
                V[2 * p + gk, col] = 1
                W[gk * n + i, col] = 1
                col += 1
    
    # Verify the full decomposition
    def verify_full(U, V, W, n, m, p, tol=1e-12):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(s - 1) > tol:
                        return False
        return True

    assert verify_full(U, V, W, n, m, p), "Full decomposition verification failed"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
