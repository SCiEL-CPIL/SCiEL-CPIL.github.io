import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to construct a lower rank decomposition
    # Hypothesis: R(1, 2, p) = 2p - 1
    # For n=3, m=4, p=5:
    # Split m=4 into two m'=2 blocks.
    # Rank = 2 * (2 * R(1, 2, p) )? No.
    # R(3, 2, 5) = 3 * R(1, 2, 5).
    # If R(1, 2, 5) = 9, then R(3, 2, 5) = 27.
    # Total rank = 27 + 27 = 54.
    
    result = _try_low_rank_decomposition(n, m, p)
    if result is not None:
        best_rank = result["rank"]
        best_result = result
        return best_result
        
    return best_result

def _try_low_rank_decomposition(n, m, p):
    # Only works for specific dimensions where we have a known construction
    if n != 3 or m != 4 or p != 5:
        return None
        
    start_time = time.time()
    
    # Construct decomposition for 1x2x5 with rank 9
    # Then embed into 3x2x5 (rank 27)
    # Then double for 3x4x5 (rank 54)
    
    U_125, V_125, W_125 = _construct_1x2x5()
    if U_125 is None:
        return None
        
    # Verify 1x2x5
    if not _verify(U_125, V_125, W_125, 1, 2, 5):
        return None
        
    # Build 3x2x5 by repeating for each of the 3 rows of A
    rank_125 = U_125.shape[1]
    rank_325 = 3 * rank_125
    
    U_325 = np.zeros((3 * 2, rank_325))
    V_325 = np.zeros((2 * 5, rank_325))
    W_325 = np.zeros((5 * 3, rank_325))
    
    for i in range(3):
        # U indices: i*m + j -> i*2 + j
        # V indices: j*p + k -> j*5 + k
        # W indices: k*n + i -> k*3 + i
        
        # U part
        for j in range(2):
            for r in range(rank_125):
                U_325[i * 2 + j, i * rank_125 + r] = U_125[j, r]
                
        # V part (same for all i, but offset in rank)
        for j in range(2):
            for k in range(5):
                for r in range(rank_125):
                    V_325[j * 5 + k, i * rank_125 + r] = V_125[j * 5 + k, r] # Note: V_125 index is j*5+k? No, V_125 shape is (2*5, rank).
                    # V_125[j*5+k, r] is correct.
                    
        # W part
        for k in range(5):
            for r in range(rank_125):
                W_325[k * 3 + i, i * rank_125 + r] = W_125[k, r] # W_125 index is k? No, W_125 shape is (5*1, rank) = (5, rank). So W_125[k, r].
                
    if not _verify(U_325, V_325, W_325, 3, 2, 5):
        return None
        
    # Build 3x4x5 by summing two 3x2x5 blocks
    rank_345 = 2 * rank_325
    
    U_345 = np.zeros((3 * 4, rank_345))
    V_345 = np.zeros((4 * 5, rank_345))
    W_345 = np.zeros((5 * 3, rank_345))
    
    # Block 1: j=0,1
    U_345[:, 0:rank_325] = U_325[:, 0:rank_325] # U_325 is (6, 27). U_345 is (12, 54).
    # Wait, U_325 has indices for j=0,1.
    # In U_345, j=0,1 are the first 6 rows.
    U_345[0:6, 0:rank_325] = U_325
    # V_345: j=0,1 rows are 0:10.
    V_345[0:10, 0:rank_325] = V_325
    # W_345: same.
    W_345[:, 0:rank_325] = W_325
    
    # Block 2: j=2,3
    # Map j=2,3 in 3x4x5 to j=0,1 in 3x2x5
    U_345[6:12, rank_325:rank_345] = U_325
    V_345[10:20, rank_325:rank_345] = V_325
    W_345[:, rank_325:rank_345] = W_325
    
    if not _verify(U_345, V_345, W_345, 3, 4, 5):
        return None
        
    return {
        "U": U_345.tolist(),
        "V": V_345.tolist(),
        "W": W_345.tolist(),
        "rank": rank_345
    }

def _construct_1x2x5():
    # Rank 9 decomposition for 1x2x5
    # C_k = A_1 B_{1k} + A_2 B_{2k}
    # Use 2p-1 = 9 terms.
    # P_1 = A_1 (B_{11} + B_{12})
    # P_2 = A_2 (B_{22} + B_{23})
    # ...
    # This construction is non-trivial.
    # If I cannot guarantee correctness, return None.
    return None, None, None

def _verify(U, V, W, n, m, p):
    # Sample verification
    np.random.seed(0)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(U.shape[1]):
                sum_U = 0.0
                for j in range(m):
                    sum_U += U[i * m + j, r] * A[i, j]
                sum_V = 0.0
                for j in range(m):
                    sum_V += V[j * p + k, r] * B[j, k]
                val += W[k * n + i, r] * sum_U * sum_V
            C_comp[i, k] = val
            
    if not np.allclose(C_true, C_comp, atol=1e-9):
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    return _verify(U, V, W, n, m, p)
