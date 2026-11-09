import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Pre-compute equation indices for verification
    num_eqs = n * n * p * p * m * m
    
    I = np.zeros(num_eqs, dtype=int)
    K = np.zeros(num_eqs, dtype=int)
    I2 = np.zeros(num_eqs, dtype=int)
    K2 = np.zeros(num_eqs, dtype=int)
    J = np.zeros(num_eqs, dtype=int)
    J2 = np.zeros(num_eqs, dtype=int)
    
    idx = 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            I[idx] = i
                            K[idx] = k
                            I2[idx] = i2
                            K2[idx] = k2
                            J[idx] = j
                            J2[idx] = j2
                            idx += 1
                            
    b = np.zeros(num_eqs)
    mask = (I2 == I) & (K2 == K) & (J == J2)
    b[mask] = 1.0
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        R = U.shape[1]
        U_vals = U[I2*m + J, :]  # shape (num_eqs, R)
        V_vals = V[J2*p + K2, :]  # shape (num_eqs, R)
        W_vals = W[K*n + I, :]  # shape (num_eqs, R)
        
        # For each equation, check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == b
        # We need to check: for each (i,k,i2,k2,j,j2), sum_r U[i2,j,r]*V[j2,k2,r]*W[k2,i2,r] = delta(i,i2)*delta(k,k2)*delta(j,j2)
        # Actually the tensor is T[i,j,k] = C[i,k] where C = A*B
        # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid product term
        # Wait, the matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor)
        # No, that's not right. Let me reconsider.
        
        # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k
        # But actually the tensor is defined such that <T, A ⊗ B> = C
        # In the decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
        # And <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_k C[i,k] * ... 
        # Actually: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j]*B[j,k]) * 1 = sum_{i,k} C[i,k]
        # So T[i,j,k] = 1 for all i,j,k? No, that gives sum_{i,k} C[i,k] which is the sum of all entries.
        
        # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) and W[k*n+i, r] is the entry.
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need T[i,j,k] = 1 for all i,j,k? That would make it a rank-1 tensor.
        
        # Actually for matrix multiplication, the tensor is:
        # T[i,j,k] = 1 if the indices form a valid term in the multiplication
        # For C = A*B: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # So T is the all-ones tensor of shape (n,m,p)? No, that's not right either.
        
        # Let me think again. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # This can be written as: C = sum_{i,k} C[i,k] * E_{i,k}
        # where E_{i,k} is the matrix with 1 at position (i,k)
        # And C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T satisfies: C[i,k] = sum_{i',j,k'} T[i',j,k'] * A[i',j] * B[j,k'] * delta(i',i) * delta(k',k)
        # Hmm, this is getting confusing.
        
        # Let me just use the verification from the original code:
        # The equations are: for all (i,k,i2,k2,j,j2):
        # sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k2*n+i2, r] = delta(i,i2)*delta(k,k2)*delta(j,j2)
        # This comes from the requirement that the decomposition reproduces the matmul tensor.
        
        prod = np.zeros((num_eqs, R))
        for r in range(R):
            prod[:, r] = U_vals[:, r] * V_vals[:, r] * W_vals[:, r]
        
        # Sum over r for each equation
        lhs = prod.sum(axis=1)
        
        return np.allclose(lhs, b, atol=1e-9)
    
    def verify_fraction(U, V, W):
        """Verify using exact fraction arithmetic."""
        R = U.shape[1]
        # Convert to fractions
        U_f = [[Fraction(int(x)) for x in row] for row in U]
        V_f = [[Fraction(int(x)) for x in row] for row in V]
        W_f = [[Fraction(int(x)) for x in row] for row in W]
        
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for k2 in range(p):
                        for j in range(m):
                            for j2 in range(m):
                                expected = 1 if (i == i2 and k == k2 and j == j2) else 0
                                actual = Fraction(0)
                                for r in range(R):
                                    actual += U_f[i2*m+j][r] * V_f[j2*p+k2][r] * W_f[k2*n+i2][r]
                                if actual != expected:
                                    return False
        return True
    
    # Column elimination strategy:
    # Start with schoolbook rank-84 solution
    # Try to eliminate columns by expressing them as linear combinations of others
    
    U_cur = U_sb.copy()
    V_cur = V_sb.copy()
    W_cur = W_sb.copy()
    R_cur = rank_sb
    
    # For each column r, try to see if it can be expressed as a combination of others
    # The condition for column r to be eliminable: there exist scalars alpha_s such that
    # U[:,r] = sum_{s != r} alpha_s * U[:,s]
    # V[:,r] = sum_{s != r} alpha_s * V[:,s]  
    # W[:,r] = sum_{s != r} alpha_s * W[:,s]
    # This is a system of linear equations.
    
    def try_eliminate_column(U, V, W, R, col_to_remove):
        """Try to eliminate column col_to_remove. Returns (success, new_U, new_V, new_W)"""
        if R <= 1:
            return False, None, None, None
        
        # Build the system: for all entries, U[:,col] = sum_{s != col} alpha_s * U[:,s]
        # This is a linear system in alpha.
        
        # Number of unknowns: R-1
        # Number of equations: (n*m + m*p + p*n) * (R-1) potentially, but we can use just enough
        
        # Actually, we need the same alpha to work for U, V, and W simultaneously.
        # So we have a joint constraint.
        
        # Let's set up: find alpha such that:
        # U[:,col] = U[:,others] @ alpha
        # V[:,col] = V[:,others] @ alpha
        # W[:,col] = W[:,others] @ alpha
        
        # This is an overdetermined system. We can solve it least-squares and check residual.
        
        others = [s for s in range(R) if s != col_to_remove]
        
        U_others = U[:, others]  # (n*m, R-1)
        V_others = V[:, others]  # (m*p, R-1)
        W_others = W[:, others]  # (p*n, R-1)
        
        U_target = U[:, col_to_remove]
        V_target = V[:, col_to_remove]
        W_target = W[:, col_to_remove]
        
        # Stack the system
        # [U_others; V_others; W_others] @ alpha = [U_target; V_target; W_target]
        
        A_mat = np.vstack([U_others, V_others, W_others])  # ((n*m+m*p+p*n), R-1)
        b_vec = np.concatenate([U_target, V_target, W_target])
        
        # Solve least squares
        try:
            alpha, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return False, None, None, None
        
        if len(residuals) > 0 and residuals[0] > 1e-9:
            return False, None, None, None
        
        # Check if alpha is close to integer or half-integer
        alpha2 = alpha * 2
        alpha_rounded = np.round(alpha2)
        if not np.allclose(alpha2, alpha_rounded, atol=1e-9):
            return False, None, None, None
        
        # Create new matrices without the column
        new_U = np.delete(U, col_to_remove, axis=1)
        new_V = np.delete(V, col_to_remove, axis=1)
        new_W = np.delete(W, col_to_remove, axis=1)
        
        return True, new_U, new_V, new_W
    
    # Iteratively try to eliminate columns
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        
        eliminated = False
        for col in range(R_cur):
            if time.time() > deadline - 10.0:
                break
            
            success, new_U, new_V, new_W = try_eliminate_column(U_cur, V_cur, W_cur, R_cur, col)
            if success:
                U_cur, V_cur, W_cur = new_U, new_V, new_W
                R_cur -= 1
                eliminated = True
                if R_cur < best_rank:
                    best_rank = R_cur
                    best_U, best_V, best_W = U_cur, V_cur, W_cur
                break  # Restart column search after elimination
        
        if not eliminated:
            break
    
    # Also try random search for better results
    for R in range(60, min(84, best_rank)):
        if time.time() > deadline - 5.0:
            break
            
        max_trials = 20
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(trial * 12345 + R)
            
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            
            U_vals = U[I2*m + J, :]
            V_vals = V[J2*p + K2, :]
            Prod = U_vals * V_vals
            
            A = np.zeros((num_eqs, (p*n)*R))
            
            r_idx = np.arange(R)
            base_cols = K * n + I
            Cols = base_cols[:, None] + r_idx[None, :] * (p*n)
            
            rows_flat = np.repeat(np.arange(num_eqs), R)
            cols_flat = Cols.flatten()
            vals_flat = Prod.flatten()
            
            A[rows_flat, cols_flat] = vals_flat
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((p*n, R))
                
                if R < best_rank:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    found = True
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
