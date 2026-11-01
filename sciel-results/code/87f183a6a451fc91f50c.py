"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank_13_search():
        """
        Attempt to find a rank-13 decomposition by sharing one term between blocks.
        We assume the structure is still close to two Strassen blocks, but one term
        from the second block is a linear combination of terms from the first block.
        This is a heuristic search.
        """
        # We'll try to solve for a rank-13 decomposition using a simplified ALS-like approach
        # or by checking specific algebraic identities.
        # Given the complexity, we will first try to see if the 14 terms are linearly dependent
        # in a way that allows removal.
        
        # For now, we return None to fall back to rank 14, as a robust rank-13 finder
        # is complex and might exceed time limits without success.
        # However, we can try a few random perturbations and local optimizations.
        
        # Let's try a simple check: Is there a linear relation among the 14 rank-1 tensors?
        # If yes, we can drop one.
        
        # Construct the 14 rank-1 tensors from the known rank-14 solution
        R = 14
        U14 = np.zeros((4, R))
        V14 = np.zeros((8, R))
        W14 = np.zeros((8, R))
        
        # Fill U14, V14, W14 with the known solution
        # (Copy from try_strassen_rectangular logic)
        U14[0,0]=1; U14[3,0]=1
        V14[0,0]=1; V14[5,0]=1
        W14[0,0]=1; W14[3,0]=1
        
        U14[2,1]=1; U14[3,1]=1
        V14[0,1]=1
        W14[1,1]=1; W14[3,1]=-1
        
        U14[0,2]=1
        V14[1,2]=1; V14[5,2]=-1
        W14[2,2]=1; W14[3,2]=1
        
        U14[3,3]=1
        V14[4,3]=1; V14[0,3]=-1
        W14[0,3]=1; W14[1,3]=1
        
        U14[0,4]=1; U14[1,4]=1
        V14[5,4]=1
        W14[0,4]=-1; W14[2,4]=1
        
        U14[2,5]=1; U14[0,5]=-1
        V14[0,5]=1; V14[1,5]=1
        W14[3,5]=1
        
        U14[1,6]=1; U14[3,6]=-1
        V14[4,6]=1; V14[5,6]=1
        W14[0,6]=1
        
        U14[0,7]=1; U14[3,7]=1
        V14[2,7]=1; V14[7,7]=1
        W14[4,7]=1; W14[7,7]=1
        
        U14[2,8]=1; U14[3,8]=1
        V14[2,8]=1
        W14[5,8]=1; W14[7,8]=-1
        
        U14[0,9]=1
        V14[3,9]=1; V14[7,9]=-1
        W14[6,9]=1; W14[7,9]=1
        
        U14[3,10]=1
        V14[6,10]=1; V14[2,10]=-1
        W14[4,10]=1; W14[5,10]=1
        
        U14[0,11]=1; U14[1,11]=1
        V14[7,11]=1
        W14[4,11]=-1; W14[6,11]=1
        
        U14[2,12]=1; U14[0,12]=-1
        V14[2,12]=1; V14[3,12]=1
        W14[7,12]=1
        
        U14[1,13]=1; U14[3,13]=-1
        V14[6,13]=1; V14[7,13]=1
        W14[4,13]=1
        
        # Flatten the 14 tensors
        tensors = []
        for r in range(R):
            t = np.zeros((4, 8, 8))
            for i in range(4):
                for j in range(8):
                    for k in range(8):
                        # Map indices to original i,j,k
                        # U index: i (0-3) -> row i
                        # V index: j (0-7) -> col j
                        # W index: k (0-7) -> output k
                        # Wait, the tensor is T_{i,j,k} = U_i V_j W_k
                        # Here U is 4x14, V is 8x14, W is 8x14
                        # So T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
                        # For a single rank-1 term r, T_r[i, j, k] = U[i,r] V[j,r] W[k,r]
                        pass # This loop is inefficient, let's use outer products
            t = np.einsum('ir,jr,kr->ijk', U14[:, r], V14[:, r], W14[:, r])
            tensors.append(t.flatten())
        
        tensors = np.array(tensors)
        
        # Check rank of the matrix of tensors (16x14)
        # If rank < 14, there is a linear dependence.
        # But we need the dependence to be such that the sum of the 14 terms is T_mult.
        # If T_mult is in the span of 13 of them, we can drop the 14th.
        
        # Let's check if any 13 of the 14 tensors span T_mult.
        # T_mult is the sum of all 14.
        T_mult = tensors.sum(axis=0)
        
        for r_drop in range(14):
            if time.time() > deadline - 10:
                break
            # Keep all except r_drop
            idx = [i for i in range(14) if i != r_drop]
            T_keep = tensors[idx] # 13x16
            
            # Check if T_mult is in the span of T_keep
            # Solve T_keep^T x = T_mult
            try:
                # Use least squares
                coeffs, residuals, rank, sv = np.linalg.lstsq(T_keep.T, T_mult, rcond=None)
                if residuals.size > 0:
                    res = residuals[0]
                    if res < 1e-10:
                        # We found a rank-13 decomposition!
                        # The coefficients are not 1, so we need to scale the terms.
                        # New term r = sum_{j != r_drop} coeffs[j] * T_j
                        # This changes the factors.
                        # We need to reconstruct U, V, W.
                        # This is hard because the linear combination of rank-1 tensors
                        # is not necessarily rank-1.
                        # However, if the dependence is simple (e.g., T_13 = T_0 + T_1),
                        # we might be able to merge them.
                        # For now, we just note that a rank reduction is possible
                        # if the linear combination preserves the rank-1 structure,
                        # which is rare.
                        pass
            except Exception:
                pass
        
        return None

    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    # Try the structured rank-14 decomposition first
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
