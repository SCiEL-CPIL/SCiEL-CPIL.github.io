"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Start with schoolbook decomposition of rank 45
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] should equal W[k*n+i, c]... 
        # Actually, the decomposition means: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = delta... 
        # No wait. The tensor decomposition for matrix multiplication is:
        # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # The rank-r decomposition says this can be computed as:
        # (A*B)_{i,k} = sum_c (sum_j U_{i*m+j, c} * A_{i,j}) * (sum_k V_{j*p+k, c} * B_{j,k}) ... 
        # Actually, the standard form is:
        # There exist rank-r tensors such that:
        # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{i,k}^{(c)}... 
        # Let me re-read the problem statement.
        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # The matrix multiplication tensor T has indices (i,j,k) where T[i,j,k] = 1 if... 
        # Actually for matrix multiplication, the tensor is:
        # T[i,j,k] = delta_{i,j,k}... no.
        # The multiplication tensor for (n x m) * (m x p) -> (n x p) is:
        # M = sum_{i,j,k} e_{i,k} ⊗ e_{i,j} ⊗ e_{j,k}
        # So the tensor T has components T[i,j,k] where i is row of A, j is column of A/row of B, k is column of B.
        # The decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c has shape (n*m), v_c has shape (m*p), w_c has shape (n*p)
        # And T[i*m+j, j*p+k, k*n+i]... 
        # Wait, the problem says W[k*n+i, c]. So W has shape (p*n, rank).
        # The tensor component at (i, j, k) is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]... 
        # No, that's not right either. For a rank-1 decomposition of a 3-tensor:
        # T_{abc} = sum_c u_a^{(c)} * v_b^{(c)} * w_c^{(c)}
        # For matrix multiplication, T_{i,j,k} = 1 if the triple (i,j,k) corresponds to a term in the product.
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # No! That's not right. The tensor for matrix multiplication is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  ... no.
        # 
        # Let me think again. The matrix multiplication map is:
        # A (n x m) * B (m x p) = C (n x p)
        # C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # This is a bilinear map. The associated tensor T has T[i,j,k] such that:
        # C_{i,k} = sum_{j} T[i,j,k] * A_{i,j} * B_{j,k}
        # So T[i,j,k] = 1 for all i,j,k.
        # 
        # A rank-r decomposition of T means:
        # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that would give sum over c of products of three factors. But T[i,j,k] is just 1.
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        # 
        # In the schoolbook case: U[i*m+j, c] = 1 if c = i*m+j + j*p+k... no.
        # In the schoolbook: for each (i,j,k), we have a term c where:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # And all other entries are 0.
        # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (the term where all three are 1).
        # Yes! That's correct.
        
        # So verification: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        
        # Efficient check using matrix operations:
        # For fixed j: sum_c U[:, j] * V[j, :] * W[:, j]... no, this is element-wise.
        # Actually, let's just do it directly for small dimensions.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_reduce(U, V, W, max_iters=200):
        """Try to reduce rank by zeroing out columns and adjusting others."""
        current_rank = U.shape[1]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 1:
                break
            
            # Try removing each column
            best_col = -1
            best_residual = float('inf')
            
            for c in range(current_rank):
                if time.time() > deadline - 5.0:
                    break
                
                # Remove column c
                mask = [i for i in range(current_rank) if i != c]
                U_sub = U[:, mask]
                V_sub = V[:, mask]
                W_sub = W[:, mask]
                
                # Check if we can adjust the remaining columns to compensate
                # This is a linear system: for each (i,j,k), 
                # sum_{c' in mask} U_sub[i*m+j, c'] * V_sub[j*p+k, c'] * W_sub[k*n+i, c'] = 1
                # This is nonlinear in general, but if we fix two factors, it's linear in the third.
                
                # Try fixing U and V, solving for W:
                # For each (i,j,k): sum_{c'} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1
                # This is linear in W[k*n+i, c'] for fixed (i,j,k), but W[k*n+i, c'] appears in multiple equations.
                
                # Actually, for fixed U and V, the constraint on W is:
                # For each k: sum_{c'} (sum_j U[:, j] * V[j, c']) * W[k*n+i, c'] ... 
                # This is getting complicated. Let me try a simpler approach.
                
                # Simple check: if removing column c still gives a valid decomposition (with the same remaining columns)
                if verify(U_sub, V_sub, W_sub, tol=1e-8):
                    best_col = c
                    best_residual = 0.0
                    break
            
            if best_col >= 0:
                mask = [i for i in range(current_rank) if i != best_col]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                current_rank = U.shape[1]
                continue
            
            # Try swapping columns to see if a different ordering helps
            # Or try setting one column to zero and adjusting others
            # For now, if we can't remove any column, break
            break
        
        return U, V, W, current_rank
    
    # Try to reduce the rank
    U, V, W, final_rank = try_reduce(U, V, W, max_iters=100)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
