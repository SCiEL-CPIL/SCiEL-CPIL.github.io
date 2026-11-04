"""Rewritten program for matrix multiplication 3x5x6 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _construct_schoolbook(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 for all valid indices, and
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_block_partition(n, m, p, deadline):
    """
    Try to decompose the n x m x p matrix multiplication tensor using a
    block-partitioning strategy.
    
    For 3x5x6: partition the 3 rows into 1+2, the 5 columns into 1+4, 
    and the 6 output columns into 1+5.
    
    This gives:
    - A 1x1x1 scalar multiplication (rank 1)
    - Cross terms and a 2x4x5 sub-problem
    
    The 2x4x5 tensor can potentially be decomposed with rank ~17-18
    using Strassen-like techniques.
    """
    if time.time() > deadline - 5:
        return None
    
    # For 3x5x6, try partitioning:
    # Row partition: 1 + 2
    # Col partition: 1 + 4  
    # Out partition: 1 + 5
    
    # The tensor decomposes as:
    # T = T_111 + T_112 + T_121 + T_122 + T_211 + T_212 + T_221 + T_222
    # where T_abc is the sub-tensor for block (a,b,c)
    
    # T_111: 1x1x1, rank 1
    # T_112: 1x1x5, rank 5 (each output is independent)
    # T_121: 1x4x1, rank 4
    # T_122: 1x4x5, rank 4*5=20 (but can be lower)
    # T_211: 2x1x1, rank 2
    # T_212: 2x1x5, rank 10
    # T_221: 2x4x1, rank 8
    # T_222: 2x4x5, this is the key sub-problem
    
    # Total naive: 1+5+4+20+2+10+8+rank(2x4x5)
    # If rank(2x4x5) <= 17, total <= 67
    # If rank(2x4x5) <= 10, total <= 60
    
    # Actually, let me think more carefully. The block partition gives:
    # (A1 A2; A3) * (B1 B2 B3; B4 B5 B6 B7 B8; ...) 
    # This is getting complicated. Let me try a different approach.
    
    # Try to find a low-rank decomposition of the 2x4x5 tensor using
    # random initialization and optimization
    
    n2, m2, p2 = 2, 4, 5
    
    # Known: rank(2x2x2) <= 7 (Strassen)
    # For 2x4x5, we can use block structure
    # 2x4x5 = 2x(2+2)x(2+2+1)
    # This gives sub-problems: 2x2x2 (rank 7), 2x2x2 (rank 7), 2x2x1 (rank 4),
    # 2x2x2 (rank 7), 2x2x2 (rank 7), 2x2x1 (rank 4)
    # Total: 7+7+4+7+7+4 = 36, which is worse than schoolbook 40
    # Hmm, not helpful directly.
    
    # Let me try a direct search for the 3x5x6 tensor with various ranks
    
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from the theoretical lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But for matrix multiplication, better bounds exist
    
    np.random.seed(42)
    random.seed(42)
    
    for target_rank in range(28, 45):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Use ALS (Alternating Least Squares) style optimization
            # For each factor, fix the other two and solve the linear system
            
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is: U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:] elementwise summed over k) = 1
                # Actually, for fixed j, and for each k:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Let M_k[r] = V[j*p+k,r] * W[k*n+i,r]
                # Then U[i*m+j,:] * M_k = 1 for all k
                # This is a linear system: sum over k of (U * M_k) = 1... no
                # It's: for each k, U dot M_k = 1
                # So we have p equations in target_rank unknowns for each (i,j)
                
                # Let's build the system for U[i*m+j, :]
                # For fixed i, j:
                # For each k in range(p): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is a linear system: A_U @ x = b, where
                # A_U[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = 1
                
                for i in range(n):
                    for j in range(m):
                        A_U = np.zeros((p, target_rank))
                        for k in range(p):
                            A_U[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_U = np.ones(p)
                        # Solve least squares
                        try:
                            U[i * m + j, :], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A_V = np.zeros((n, target_rank))
                        for i in range(n):
                            A_V[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_V = np.ones(n)
                        try:
                            V[j * p + k, :], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A_W = np.zeros((m, target_rank))
                        for j in range(m):
                            A_W[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_W = np.ones(m)
                        try:
                            W[k * n + i, :], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                        except:
                            pass
            
            # Check if converged
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None and best_rank < n * m * p:
            break
    
    return best_result


def _try_strassen_like(n, m, p, deadline):
    """
    Try to use Strassen-like recursive decomposition.
    
    For 3x5x6, we can pad to 4x6x6 or use a different strategy.
    
    Actually, let's try a direct approach: use the fact that
    3x5x6 can be embedded in a 4x6x6 problem, and use known
    decompositions.
    """
    if time.time() > deadline - 5:
        return None
    
    # Try a structured decomposition based on Kronecker products
    # and known low-rank decompositions of smaller tensors
    
    # For now, return None to fall back to schoolbook
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as safe fallback
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try block partitioning strategy
    result = _try_block_partition(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0]) if result["U"] and result["U"][0] else 0
        if rank < best_rank:
            best_rank = rank
            best_result = result
    
    # Try Strassen-like approach
    result = _try_strassen_like(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0]) if result["U"] and result["U"][0] else 0
        if rank < best_rank:
            best_rank = rank
            best_result = result
    
    return best_result

# EVOLVE-BLOCK-END
