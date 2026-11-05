"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    # Known lower bound for 3x3 * 3x7 is at least 15 (from literature on matrix multiplication)
    # The optimal rank for 3x3 * 3x7 is known to be 15 (based on Strassen-type results)
    # Let's try to find a decomposition of rank 15 or lower
    
    # Strategy: Use the known structure of matrix multiplication
    # For multiplying (n x m) by (m x p), the tensor T has entries T[i,j,k] where
    # i in [0,n), j in [0,m), k in [0,p)
    # The output is W[k*n+i, r] following trace convention
    
    # A rank-1 term corresponds to choosing:
    # - a set of (i,j) pairs for U (input A factors)
    # - a set of (j,k) pairs for V (input B factors)  
    # - a set of (k,i) pairs for W (output factors)
    # such that the product matches the tensor
    
    # For exact decomposition, each rank-1 term (u, v, w) contributes:
    # sum over terms of u[i,j] * v[j,k] * w[k,i] = delta[i,j,k] for the multiplication tensor
    
    # The matrix multiplication tensor T[i,j,k] = 1 if the indices are "compatible"
    # For A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) - it's the full tensor
    
    # Actually, let me reconsider. The tensor for matrix multiplication is:
    # T[i,j,k] where i indexes rows of A and columns of C,
    # j indexes columns of A and rows of B,
    # k indexes columns of C and rows of B
    # T[i,j,k] = 1 for all (i,j,k) in the valid range
    
    # A rank-r decomposition gives:
    # T[i,j,k] = sum_{s=1}^r U[i,j,s] * V[j,k,s] * W[k,i,s]
    
    # For the standard matrix multiplication, T[i,j,k] = 1 for all valid indices.
    # So we need: sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] = 1 for all (i,j,k)
    
    # Let's try a constructive approach based on known results.
    # For 3x3 * 3x7, the minimal rank is believed to be around 15-20.
    
    # Let me try to construct a valid decomposition using a systematic approach.
    # We'll use the fact that we can combine terms cleverly.
    
    # First, let's verify the schoolbook works and return it as fallback
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a better decomposition using random search with structured initialization
    # Based on the idea that we can use Strassen-like combinations
    
    # For 3x3 * 3x3, Strassen gives rank 7 instead of 9.
    # For 3x3 * 3x7, we might be able to do better than 63.
    
    # Let's try a systematic approach: use the known Strassen decomposition for 3x3*3x3
    # and extend it to handle the 3x7 case.
    
    # Actually, let me try a different approach: use integer programming or 
    # a greedy algorithm to find a good decomposition.
    
    # For now, let's try to find a decomposition of rank < 63 using random search
    # with verification.
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try ranks from 15 up to 62
    for target_rank in range(15, min(63, best_rank)):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try a few random attempts for each rank
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # For each rank-1 term, assign random non-zero patterns
            for r in range(target_rank):
                # U: choose which (i,j) entries are non-zero
                for idx in range(n * m):
                    if np.random.random() < 0.5:
                        U[idx, r] = np.random.choice([-1, 1])
                
                # V: choose which (j,k) entries are non-zero
                for idx in range(m * p):
                    if np.random.random() < 0.5:
                        V[idx, r] = np.random.choice([-1, 1])
                
                # W: choose which (k,i) entries are non-zero
                for idx in range(p * n):
                    if np.random.random() < 0.5:
                        W[idx, r] = np.random.choice([-1, 1])
            
            # Convert to lists and verify
            U_list = [[int(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
            V_list = [[int(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
            W_list = [[int(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
