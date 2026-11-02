import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Vectorized verification"""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                    if val != 1:
                        return False
        return True

    def try_output_basis_change(r):
        """
        Try a linear change of basis on the output dimension.
        Instead of computing C[i,k] directly, compute D[i,k] = sum_i' M[i,i'] C[i',k]
        where M is a 2x2 invertible matrix with integer/rational entries.
        
        The tensor for D is: T_D[i,j,k] = sum_i' M[i,i'] * T[i',j,k] = M[i,i'] (since T=1)
        
        So T_D[i,j,k] = M[i,0] + M[i,1]... wait, T[i',j,k]=1 for all i', so
        T_D[i,j,k] = sum_i' M[i,i'] * 1 = M[i,0] + M[i,1]
        
        This is independent of j and k! So the tensor becomes:
        T_D[i,j,k] = s_i where s_i = M[i,0] + M[i,1]
        
        This is a rank-1 tensor in (j,k) for each i, and the i-dependence is just a scalar.
        So the rank is at most 2 (one for each i, if s_i != 0).
        
        But we need to recover C from D: C = M^{-1} D.
        
        This doesn't directly help because the tensor structure is too simple.
        
        Let me reconsider. The change of basis should be on the A dimension, not C.
        """
        if time.time() > deadline - 10:
            return None
        
        # Try: change of basis on A (input) dimension
        # A' = A * M where M is 3x3 invertible
        # Then C = A' * M^{-1} * B
        # This doesn't change the rank.
        
        # Alternative: the key is to find a decomposition where W has a special structure
        # that allows sharing between i=0 and i=1.
        
        # For each column c, suppose W[k*n+0, c] and W[k*n+1, c] are related.
        # E.g., W[k*n+1, c] = W[k*n+0, c] for all k, then the column contributes
        # equally to both output rows.
        
        # Or W[k*n+1, c] = -W[k*n+0, c], then it contributes with opposite sign.
        
        # Let's try: for each column, W has non-zeros at (k0, 0) and (k1, 1) with
        # W[k0*n+0, c] = a, W[k1*n+1, c] = b where a,b are from {-1, 1, 2, -2}
        
        for trial in range(100):
            if time.time() > deadline - 5:
                return None
            
            W = np.zeros((p_rows, r), dtype=np.int64)
            for c in range(r):
                k0 = random.randint(0, p-1)
                k1 = random.randint(0, p-1)
                a = random.choice([-1, 1, 2, -2])
                b = random.choice([-1, 1, 2, -2])
                W[k0*n + 0, c] = a
                W[k1*n + 1, c] = b
            
            # Now solve for U and V
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # 
            # Fix V sparsely: V[j*p+k, c] non-zero for at most one (j,k) per column
            V = np.zeros((m_rows, r), dtype=np.int64)
            for c in range(r):
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                V[j*p+k, c] = random.choice([-1, 1, 2, -2])
            
            # Solve for U: for each (i,j), M[i,j] @ U[i*m+j,:] = ones(p)
            # where M[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c]
            U = np.zeros((n_rows, r), dtype=np.int64)
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    # Solve M @ u = ones(p)
                    try:
                        u_float, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u_float).astype(np.int64)
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None

    def try_strassen_variant(r):
        """
        Try to improve on the Strassen-based solution by finding a better
        decomposition of the 2x2 x 2x9 sub-problems or by merging columns.
        """
        if time.time() > deadline - 10:
            return None
        
        # The current solution uses:
        # 4 * 7 = 28 for Strassen on 2x2 x 2x9 blocks (k=0..7 in pairs)
        # 4 for the last column (k=8) of B_T
        # 18 for A_R x B_B
        # Total: 50
        
        # Can we do the last column (k=8) more efficiently?
        # A_L (2x2) * B_col8 (2x1): this is a 2x2 x 2x1 problem.
        # Rank of (2,2,2) is 7 (Strassen). But (2,2,1) is simpler.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # This is 4 multiplications in schoolbook, but can we do it in 3?
        # (2,2,1) tensor: T[i,j,0] = 1. Rank is 4 (schoolbook is optimal for (2,2,1)).
        # So 4 is minimal for that part.
        
        # Can we merge the k=8 computation with the Strassen blocks?
        # Probably not directly, but let's try a different block structure.
        
        # Alternative: use a 3-column Strassen-like decomposition for the 2x3 x 3x9.
        # Split A into columns: A = [A0, A1, A2] where each is 2x1.
        # C = A0*B[:,0] + A1*B[:,1] + A2*B[:,2]
        # Each A_i * B[:,j] is a 2x9 matrix = outer product of 2x1 and 1x9.
        # This is 3 * 18 = 54 in schoolbook.
        
        # Not helpful. Let's try the output basis change idea more carefully.
        
        # Key idea: compute D[0,:] = C[0,:] + C[1,:] and D[1,:] = C[0,:] - C[1,:]
        # D[0,k] = (A[0,:] + A[1,:]) * B[:,k]
        # D[1,k] = (A[0,:] - A[1,:]) * B[:,k]
        # 
        # Let A_plus = A[0,:] + A[1,:] (1x3) and A_minus = A[0,:] - A[1,:] (1x3)
        # D[0,:] = A_plus * B (1x3 * 3x9)
        # D[1,:] = A_minus * B (1x3 * 3x9)
        #
        # Each is a (1,3,9) tensor with rank 27 in schoolbook.
        # Total: 54. Worse than 50.
        
        # What if we use Strassen on the 2x2 part and handle the 2x1 part jointly?
        # This is what the current solution does.
        
        # Let's try: can we reduce the 18 for A_R x B_B?
        # A_R is 2x1, B_B is 1x9. C_R = A_R * B_B is 2x9.
        # C_R[i,k] = A_R[i,0] * B_B[0,k]
        # This is a rank-1 matrix, but in tensor decomposition, each (i,k) entry
        # requires a separate multiplication unless we can share.
        #
        # T_R[i,0,k] = 1 for i in {0,1}, k in {0..8}
        # This is a (2,1,9) tensor. Rank is 18 (schoolbook).
        # Can we do better? The (2,1,9) tensor has rank 18 (it's essentially 18 independent
        # bilinear forms).
        
        # So the minimum seems to be around 50. Let's verify with a more careful search.
        
        # Try: reduce the Strassen part by using a different decomposition for 2x2 x 2x9.
        # The (2,2,9) tensor... what's its rank?
        # (2,2,n) tensor: for n=1, rank=4. For n=2, rank=7 (Strassen).
        # For n=9, the rank is at most 4*9=36 (schoolbook) and at most 7*ceil(9/2)+4 = 7*5+4=39
        # using Strassen on pairs plus schoolbook on the last one.
        # Actually, 7*4 + 4 = 32 for 4 pairs + 1 single.
        
        # Wait, the current solution does 4 Strassen blocks (7 each) = 28 for k=0..7,
        # plus 4 for k=8. Total 32 for the 2x2 x 2x9 part.
        # Plus 18 for 2x1 x 1x9. Total 50.
        
        # Can we do the 2x2 x 2x9 in less than 32?
        # The (2,2,9) tensor rank... I'm not sure of the exact value.
        # Let's assume it's 32 and focus on the 2x1 x 1x9 part.
        
        # Actually, wait. The A matrix is 2x3, not 2x2 + 2x1 in a way that separates cleanly.
        # A = [A_L | A_R] where A_L is 2x2 and A_R is 2x1.
        # B = [B_T; B_B] where B_T is 2x9 and B_B is 1x9.
        # C = A_L * B_T + A_R * B_B
        
        # The tensor is: T[i,j,k] = 1 for all i,j,k.
        # This is the sum of two tensors: T1[i,j,k] = 1 if j<2 else 0 (for A_L*B_T)
        # and T2[i,j,k] = 1 if j=2 else 0 (for A_R*B_B).
        
        # The rank of the sum is at most the sum of ranks, but can be less.
        
        # Let's just try a more aggressive random search.
        return None

    def try_aggressive_search(r, num_trials=200):
        """
        More aggressive random search with better structure.
        """
        if time.time() > deadline - 10:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 3:
                break
            
            # W structure: each column has exactly 2 non-zeros, one for i=0, one for i=1
            W = np.zeros((p_rows, r), dtype=np.int64)
            for c in range(r):
                k0 = random.randint(0, p-1)
                k1 = random.randint(0, p-1)
                a = random.choice([-1, 1, 2, -2, 3, -3])
                b = random.choice([-1, 1, 2, -2, 3, -3])
                W[k0*n + 0, c] = a
                W[k1*n + 1, c] = b
            
            # V structure: each column has 1 non-zero
            V = np.zeros((m_rows, r), dtype=np.int64)
            for c in range(r):
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                V[j*p+k, c] = random.choice([-1, 1, 2, -2])
            
            # Solve for U
            U = np.zeros((n_rows, r), dtype=np.int64)
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    try:
                        u_float, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u_float).astype(np.int64)
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None

    # Try to find lower rank solutions
    best_result = None
    
    # Try aggressive search for ranks 49 down to 42
    for r in range(49, 41, -1):
        if time.time() > deadline - 20:
            break
        res = try_aggressive_search(r, num_trials=100)
        if res:
            best_result = res
            break
    
    # Try output basis change approach
    if best_result is None and time.time() < deadline - 15:
        for r in range(49, 44, -1):
            if time.time() > deadline - 5:
                break
            res = try_output_basis_change(r)
            if res:
                best_result = res
                break
    
    # Fallback to rank 50
    if best_result is None:
        best_result = current_solution()
    
    return best_result
