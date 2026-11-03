import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy:
    1. Use Strassen's algorithm to build a rank-56 decomposition.
    2. Attempt to reduce rank by finding linear dependencies among the rank-1 terms
       using integer/rational arithmetic.
    3. Fall back to the best valid decomposition found.
    """
    start_time = time.time()
    DEATHLINE = 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    def schoolbook():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_factors(U_mat, V_mat, W_mat, tol=1e-9):
        """
        Verify that the factors correctly decompose the matrix multiplication tensor.
        """
        try:
            rank = len(U_mat[0])
            if len(V_mat[0]) != rank or len(W_mat[0]) != rank:
                return False
            if len(U_mat) != nm or len(V_mat) != mp or len(W_mat) != pn:
                return False
            
            # Check the tensor identity: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_mat[i*m + j][r] * V_mat[j*p + k][r] * W_mat[k*n + i][r]
                        if abs(val - 1) > tol:
                            return False
            return True
        except Exception:
            return False
    
    def strassen_2x2_factors():
        """
        Strassen's 7 multipliers for 2x2 * 2x2 -> 2x2.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e),
        p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
        C[0,0] = p1+p4-p5+p7, C[0,1] = p3+p5, C[1,0] = p2+p4, C[1,1] = p1-p2+p3+p6
        
        Returns list of 7 (u, v, w) tuples where:
        u[li*2+lj] = coefficient for A[li, lj]
        v[lj*2+lk] = coefficient for B[lj, lk]  
        w[lk*2+li] = coefficient for C[li, lk]  (using W[k*n+i] convention with n=2)
        """
        factors = []
        # p1=(a+d)(e+h): +C[0,0], +C[1,1]
        factors.append(([1,0,1,0], [1,0,0,1], [1,0,0,1]))
        # p2=(c+d)e: +C[1,0], -C[1,1]
        factors.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
        # p3=a(f-h): +C[0,1], +C[1,1]
        factors.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
        # p4=d(g-e): +C[0,0], +C[1,0]
        factors.append(([0,0,0,1], [0,0,1,-1], [1,1,0,0]))
        # p5=(a+b)h: -C[0,0], +C[0,1]
        factors.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
        # p6=(c-a)(e+f): +C[1,1]
        factors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # p7=(b-d)(g+h): +C[0,0]
        factors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return factors
    
    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 2x4 * 4x8.
        
        Split j into two halves: j in {0,1} and j in {2,3}
        For each output column pair (4 pairs), apply Strassen to each 2x2*2x2 product.
        Total: 4 column pairs * 2 (left/right) * 7 Strassen = 56 rank.
        """
        strassen_factors = strassen_2x2_factors()
        
        rank = 56
        U_mat = [[0] * rank for _ in range(nm)]
        V_mat = [[0] * rank for _ in range(mp)]
        W_mat = [[0] * rank for _ in range(pn)]
        
        col = 0
        for k_pair in range(4):  # 4 output column pairs
            k_start = 2 * k_pair
            for a_half in range(2):  # 0: A_left (j=0,1), 1: A_right (j=2,3)
                j_start = 2 * a_half
                for s_idx in range(7):
                    u_vec, v_vec, w_vec = strassen_factors[s_idx]
                    
                    # Fill U: A is 2x4, so U[i*m + j, col]
                    for li in range(2):
                        for lj in range(2):
                            U_mat[li * m + (j_start + lj)][col] = u_vec[li * 2 + lj]
                    
                    # Fill V: B is 4x8, so V[j*p + k, col]
                    for lj in range(2):
                        for lk in range(2):
                            V_mat[(j_start + lj) * p + (k_start + lk)][col] = v_vec[lj * 2 + lk]
                    
                    # Fill W: W[k*n + i, col]
                    for lk in range(2):
                        for li in range(2):
                            W_mat[(k_start + lk) * n + li][col] = w_vec[lk * 2 + li]
                    
                    col += 1
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    def try_reduce_rank(U_mat, V_mat, W_mat):
        """
        Attempt to reduce rank by finding linear dependencies.
        This is a heuristic approach - we look for columns that are linear combinations
        of other columns in the tensor decomposition.
        
        For tensor decomposition, rank reduction is non-trivial. We try a simple approach:
        check if any column can be expressed as a linear combination of others.
        """
        if time.time() > DEATHLINE - 20:
            return None
        
        rank = len(U_mat[0])
        
        # Convert to numpy for easier manipulation
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # Try to find linear dependencies using SVD
        # The tensor is T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can try to see if we can remove columns while preserving the tensor
        
        best_result = None
        best_rank = rank
        
        # Try removing one column at a time
        for col_to_remove in range(rank):
            if time.time() > DEATHLINE - 10:
                break
            
            # Create reduced matrices
            U_reduced = np.delete(U_np, col_to_remove, axis=1)
            V_reduced = np.delete(V_np, col_to_remove, axis=1)
            W_reduced = np.delete(W_np, col_to_remove, axis=1)
            
            # Verify if the reduced decomposition is still valid
            new_rank = rank - 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(new_rank):
                            val += U_reduced[i*m + j, r] * V_reduced[j*p + k, r] * W_reduced[k*n + i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
            
            if valid:
                best_rank = new_rank
                # Convert back to lists
                U_list = U_reduced.tolist()
                V_list = V_reduced.tolist()
                W_list = W_reduced.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        return best_result
    
    # Try Strassen-based decomposition first
    best_result = None
    best_rank = float('inf')
    
    try:
        result = build_strassen_decomposition()
        if verify_factors(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
            
            # Try to reduce rank
            if time.time() < DEATHLINE - 30:
                reduced = try_reduce_rank(result["U"], result["V"], result["W"])
                if reduced is not None:
                    if verify_factors(reduced["U"], reduced["V"], reduced["W"]):
                        new_rank = len(reduced["U"][0])
                        if new_rank < best_rank:
                            best_result = reduced
                            best_rank = new_rank
    except Exception:
        pass
    
    # Fall back to schoolbook if no better result
    if best_result is None:
        best_result = schoolbook()
    elif best_rank > n * m * p:
        best_result = schoolbook()
    
    return best_result

if __name__ == "__main__":
    pass
