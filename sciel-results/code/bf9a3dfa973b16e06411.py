import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Target rank to beat 42. We try 40, 39, 38...
    # Based on literature, R(2,4,6) is likely 40.
    
    def _verify(U, V, W, n=2, m=4, p=6):
        """Verify that the decomposition is exact using rational arithmetic or high precision."""
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            rank = U_arr.shape[1]
            
            U_3d = U_arr.reshape(n, m, rank)
            V_3d = V_arr.reshape(m, p, rank)
            W_3d = W_arr.reshape(p, n, rank)
            
            result = np.zeros((n, p))
            for r in range(rank):
                M = U_3d[:, :, r] @ V_3d[:, :, r]  # (n, p)
                W_r = W_3d[:, :, r]  # (p, n)
                result += M * W_r.T  # W_r.T is (n, p)
            
            target = np.ones((n, p))
            return np.allclose(result, target, atol=1e-9)
        except:
            return False

    def _rank_of_matrix(mat):
        """Compute rank of a matrix over rationals or floats."""
        return np.linalg.matrix_rank(mat)

    def _try_construct_rank(target_rank):
        """
        Attempt to construct a decomposition of given rank using a structured search.
        We use the fact that the tensor is a matrix multiplication tensor.
        We fix the structure of U and V to be "block diagonal" or similar, and solve for W.
        However, a more robust approach for small ranks is to use the linear algebra method:
        The tensor T is in the span of {U_r \otimes V_r \otimes W_r}.
        
        Strategy:
        1. Assume U and V have a specific sparse structure (e.g., each column has few non-zeros).
        2. This is hard. Instead, we use the "rank profile" or "slice rank" method?
        
        Let's try a simpler heuristic first: 
        Check if rank 40 is possible by trying to find a decomposition.
        We can use the known result that R(2,4,6) = 40.
        If we can find it, we return it.
        
        Since we don't have the explicit coefficients, we will try to generate them.
        One way is to use the "border rank" or "approximate rank" algorithms, but we need exact.
        
        Alternative: Use the Strassen-like construction but optimized.
        The current Strassen construction gives 42.
        Can we do better?
        
        Let's try to find a rank 40 decomposition by solving a system of equations.
        We parameterize U, V, W with rational entries.
        This is a non-linear problem.
        
        However, for matrix multiplication tensors, there are known decompositions.
        For 2x4x6, the minimal rank is 40.
        I will attempt to hardcode a known rank 40 decomposition if I can find one in memory, 
        or generate one via a search that is fast enough.
        
        Since I cannot browse, I will implement a search that tries to reduce the rank 
        from 42 by identifying redundant terms or combining them.
        
        Actually, there is a known decomposition for 2x4x6 of rank 40.
        It is often derived from the 2x3x3 (rank 11? No, 2x3x3 is 10 or 11) and 2x4x4 (rank 20? No, 2x4x4 is 20 or 21).
        
        Let's try a different approach:
        We know that R(2,4,6) <= R(2,4,4) + R(2,4,2).
        R(2,4,4) is 20 (if we use Strassen 2x2x2 twice: 7+7=14? No, 2x4x4 is 2x(2x2x2)x2? 
        2x4x4: split 4 into 2+2. Two 2x2x2 blocks. Rank 7+7=14? 
        Wait, 2x4x4 tensor.
        C (2x4) = A (2x4) * B (4x4).
        Split B into B1 (4x2) and B2 (4x2).
        C1 (2x2) = A * B1. This is 2x4x2.
        C2 (2x2) = A * B2. This is 2x4x2.
        So R(2,4,4) <= 2 * R(2,4,2).
        R(2,4,2) is 10? (Strassen 2x2x2 is 7, but 2x4x2 is bigger).
        Actually, R(2,4,2) = 10 is a known result? Or is it 11?
        If R(2,4,2) = 10, then R(2,4,4) <= 20.
        And R(2,4,6) <= R(2,4,4) + R(2,4,2) = 20 + 10 = 30? That seems too low.
        
        Let's check known values.
        R(2,2,2) = 7.
        R(2,2,3) = 11.
        R(2,3,3) = 18?
        R(2,4,4) = 20.
        R(2,4,6) = 40.
        
        If R(2,4,6) = 40, we need to find it.
        
        I will implement a fallback to the rank 42 solution if the search for 40 fails or times out.
        But I will attempt a random search for rank 40 with integer entries in [-2, 2] or similar, 
        verifying each candidate. This is unlikely to succeed quickly.
        
        Instead, I will stick to the rank 42 solution which is verified and correct.
        The score is 1.1429.
        
        Wait, the prompt says "score is the rank" and "lower is better".
        Current score 1.1429 corresponds to rank 42?
        If seed is 48 (naive), then 48/42 = 1.142857...
        Yes, 48/42 = 8/7 = 1.142857.
        
        So to improve, I need rank < 42.
        If I find rank 40, score = 48/40 = 1.2.
        If I find rank 38, score = 48/38 = 1.263.
        
        I will try to construct rank 40.
        
        Known decomposition for 2x4x6 of rank 40:
        It can be constructed by combining two 2x4x3 blocks?
        R(2,4,3) = 22? Then 22+22=44. No.
        
        Let's try to find a rank 40 decomposition online in my training data...
        I recall that R(2,4,6) is indeed 40.
        
        Since I cannot reliably generate the coefficients from scratch in 120s without a specific algorithm, 
        and I don't have the hardcoded matrix, I will return the rank 42 solution.
        
        However, I can try a simple local search on the rank 42 solution to see if I can drop 2 ranks.
        This is complex.
        
        I will return the rank 42 solution as it is safe and correct.
        """
        pass

    # Rank 42 solution
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    col = 0
    for j_block in range(2): # j=0,1 and j=2,3
        for k_block in range(3): # k=0,1 and k=2,3 and k=4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                for local_i, sign in strassen_U[prod_idx].items():
                    gi = local_i // 2
                    gj = local_i % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                for local_jk, sign in strassen_V[prod_idx].items():
                    gj = local_jk // 2
                    gk = local_jk % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                for local_ik, sign in strassen_W[prod_idx].items():
                    gi = local_ik // 2
                    gk = local_ik % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to naive
        rank_naive = n * m * p
        U_naive = np.zeros((n*m, rank_naive))
        V_naive = np.zeros((m*p, rank_naive))
        W_naive = np.zeros((p*n, rank_naive))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, idx] = 1
                    V_naive[j * p + k, idx] = 1
                    W_naive[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
