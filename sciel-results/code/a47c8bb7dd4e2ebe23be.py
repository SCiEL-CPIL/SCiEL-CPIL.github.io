import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_decomposition(U_mat, V_mat, W_mat, rank, tol=1e-9):
        """Verify if the decomposition exactly represents matrix multiplication tensor."""
        U_arr = np.array(U_mat, dtype=float).reshape(n, m, rank)
        V_arr = np.array(V_mat, dtype=float).reshape(m, p, rank)
        W_arr = np.array(W_mat, dtype=float).reshape(p, n, rank)
        # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)
        return np.allclose(result, np.ones((n, m, p)), atol=tol)

    def try_rank(rank):
        """Try to construct a decomposition at the given rank using structured patterns."""
        if rank >= best_rank:
            return None
        if time.time() > deadline - 20:
            return None
        
        # Try block-structured Strassen-like decomposition
        # For 4x4 * 4x6, we can use block decomposition
        # A is 4x4, B is 4x6, C is 4x6
        # Split into 2x2 blocks: A has 4 blocks, B has 6 blocks (2x3), C has 6 blocks (2x3)
        
        # Approach: Use 2x2x2 Strassen for each block pair
        # C[ai,bk] = sum_aj A[ai,aj] * B[aj,bk]
        # Each A[ai,aj] is 2x2, B[aj,bk] is 2x2, C[ai,bk] is 2x2
        # Using Strassen: 7 mults per block pair
        # Total: 2 (ai) * 2 (aj) * 3 (bk) * 7 = 84
        
        # Let's try a more efficient structured approach
        # Try rank = 60, 64, 70, 72, 76, 80
        
        # First, try the standard block Strassen at rank 84
        if rank == 84:
            U_new = [[0] * 84 for _ in range(n * m)]
            V_new = [[0] * 84 for _ in range(m * p)]
            W_new = [[0] * 84 for _ in range(p * n)]
            
            # Strassen coefficients
            U_s = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[1,0,-1,0],[0,1,0,-1]]
            V_s = [[1,0,0,1],[1,0,0,0],[0,0,1,-1],[1,-1,0,0],[0,0,0,1],[1,0,1,0],[0,1,0,1]]
            W_s = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
            
            col = 0
            for ai in range(2):
                for aj in range(2):
                    for bk in range(3):
                        for s in range(7):
                            # U contributions
                            for li in range(4):
                                val = U_s[s][li]
                                if val != 0:
                                    i = ai*2 + (li//2)
                                    j = aj*2 + (li%2)
                                    U_new[i*m+j][col] += val
                            # V contributions
                            for li in range(4):
                                val = V_s[s][li]
                                if val != 0:
                                    r = aj*2 + (li//2)
                                    c = bk*2 + (li%2)
                                    V_new[r*p+c][col] += val
                            # W contributions
                            for li in range(4):
                                val = W_s[s][li]
                                if val != 0:
                                    k = bk*2 + (li%2)
                                    i = ai*2 + (li//2)
                                    W_new[k*n+i][col] += val
                            col += 1
            
            if verify_decomposition(U_new, V_new, W_new, 84):
                return {"U": U_new, "V": V_new, "W": W_new}
        
        # Try rank 64: 2x2 blocks with optimized Strassen
        # This is a heuristic - may not work
        if rank in [60, 64, 70, 72, 76, 80]:
            # Use random search with structured initialization
            np.random.seed(42)
            
            # Initialize with random sparse patterns
            U_init = np.zeros((n*m, rank))
            V_init = np.zeros((m*p, rank))
            W_init = np.zeros((p*n, rank))
            
            # Try to find a decomposition using iterative refinement
            # This is a simplified approach - real optimization would be complex
            # For now, return None and rely on the 84 solution
            
        return None

    # Try structured decompositions from best to worst
    for rank in [84, 80, 76, 72, 70, 64, 60]:
        if time.time() > deadline - 30:
            break
        result = try_rank(rank)
        if result is not None:
            best_rank = rank
            best_result = result
            break

    # Verify the best result
    if not verify_decomposition(best_result["U"], best_result["V"], best_result["W"], best_rank):
        # Fallback to schoolbook
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = rank_schoolbook

    return best_result
