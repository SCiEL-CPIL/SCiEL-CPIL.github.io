import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_solution(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if v is not None:
                    v2 = v * 2
                    if abs(v2 - round(v2)) > 1e-9:
                        return False
    # Verify the tensor equation: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-8:
                    return False
    return True

def _verify_solution_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m+j]  # (rank,)
            for k in range(p):
                v_row = Va[j*p+k]  # (rank,)
                w_row = Wa[k*n+i]  # (rank,)
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1) > 1e-8:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin within 150s
    
    # Schoolbook decomposition (Rank 105)
    best_solution = _build_schoolbook()
    best_rank = n * m * p  # 105
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # 35

    # Strategy:
    # 1. Check if the tensor has a lower numerical rank using SVD on unfoldings.
    # 2. If numerical rank < 105, attempt to find an exact half-integer solution for that rank.
    # 3. If not, return schoolbook.
    
    # Construct the tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k]
    # Actually, the multiplication tensor is defined by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) and value 1.
    # We are looking for U, V, W such that:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    # Note: The W index in the task is W[k*n+i, r], which corresponds to the output index (k, i) 
    # but the standard matrix multiplication output is C[i, k].
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the output matrix is indexed by k (row?) and i (col?) or similar.
    # Let's check the schoolbook:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    # Product: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Sum over r.
    # For a specific (i, j, k), only the term r = i*m*p + j*p + k is non-zero?
    # No, in schoolbook, each (i,j,k) has its own r.
    # So sum_r ... = 1.
    
    # The tensor T_{i,j,k} = 1.
    # We can unfold T to a matrix M1 of size (n*m) x p.
    # M1[i*m+j, k] = T[i, j, k] = 1.
    # Rank of M1 is min(n*m, p) = min(15, 7) = 7.
    # This doesn't directly give the tensor rank.
    # Tensor rank is the minimum R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    
    # Let's try to find a solution for Rank R = 104, 103... down to lower_bound.
    # But first, a quick check: Is the tensor "generic"?
    # If we assume the rank is 105, we return immediately.
    # To potentially improve, we can try a few random structured searches for R < 105.
    
    # Improved Search: Use sparse random U, V and solve for W.
    # We optimize the linear system construction.
    
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Limit trials for speed
        trials = 10 if R > 90 else 5
        
        for t in range(trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse U, V
            # Density 0.2
            U_rand = np.random.choice([0, 1, -1], size=(n * m, R), p=[0.8, 0.1, 0.1])
            V_rand = np.random.choice([0, 1, -1], size=(m * p, R), p=[0.8, 0.1, 0.1])
            
            # Build A matrix for W
            # Eq (i, j, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # A[eq_idx, unk_idx] = U[i*m+j, r] * V[j*p+k, r]
            # unk_idx = (k*n+i)*R + r
            
            n_eq = n * m * p
            n_unk = p * n * R
            A = np.zeros((n_eq, n_unk))
            
            # Fill A
            # Optimization: Only iterate over non-zeros in U and V
            for i in range(n):
                for j in range(m):
                    u_row = U_rand[i * m + j]
                    nz_u = np.nonzero(u_row)[0]
                    if len(nz_u) == 0: continue
                    for k in range(p):
                        v_row = V_rand[j * p + k]
                        # Coeffs are u_row * v_row
                        # We need to add to A[eq_idx, (k*n+i)*R + r]
                        eq_idx = i * (m * p) + j * p + k
                        # Only r where u_row[r] != 0 AND v_row[r] != 0
                        # Intersect nz_u and nz_v
                        nz_v = np.nonzero(v_row)[0]
                        if len(nz_v) == 0: continue
                        # Find common indices
                        # Since R is small, just loop
                        for r in nz_u:
                            if v_row[r] != 0:
                                coeff = u_row[r] * v_row[r]
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            # Solve
            try:
                b = np.ones(n_eq)
                # Use lstsq
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Round to half-integer
                w_rounded = np.round(w_sol * 2) / 2
                
                # Verify
                if np.allclose(A @ w_rounded, b, atol=1e-8):
                    W_list = [[0] * R for _ in range(p * n)]
                    for idx in range(n_unk):
                        ki = idx // R
                        r = idx % R
                        k_val = ki // n
                        i_val = ki % n
                        W_list[k_val * n + i_val][r] = w_rounded[idx]
                    
                    if _verify_solution_fast(U_rand.tolist(), V_rand.tolist(), W_list):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_list}
                        
            except Exception:
                continue
    
    return best_solution
# EVOLVE-BLOCK-END
