import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is 2x2, B is 2x10, C is 2x10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    
    def build_strassen_rank_35():
        """Build rank-35 decomposition using Strassen's 7-term 2x2x2 decomposition
        Kronecker-producted with the 5-term identity for the p=10 dimension."""
        R = 35
        U = np.zeros((4, R), dtype=float)
        V = np.zeros((20, R), dtype=float)
        W = np.zeros((20, R), dtype=float)
        
        # Strassen terms for 2x2x2
        # A indices: 0:00, 1:01, 2:10, 3:11
        # B indices: 0:00, 1:01, 2:10, 3:11
        # C indices: 0:00, 1:01, 2:10, 3:11 (Flattened as k*n+i)
        # Note: The standard Strassen algorithm computes C = A*B.
        # The tensor decomposition T = sum U_r \otimes V_r \otimes W_r
        # such that T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k?} no.
        # The target tensor for M_{2,2,2} is T[i,j,k] = 1 if i==k? No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor is T_{i,j,k} such that contraction with A and B gives C.
        # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # So T_{i,j,k} = 1 if the index structure matches?
        # Actually, the tensor M_{n,m,p} has entries M_{i,j,k} = 1 if i,j,k form a valid "path"?
        # No, M_{i,j,k} = 1 if the multiplication A_{i,j} B_{j,k} contributes to C_{i,k}.
        # This is always 1 for valid indices.
        # So the target tensor is T_{i,j,k} = 1 for all i,j,k?
        # No, that would be the "all ones" tensor.
        # The multiplication tensor M_{n,m,p} is defined by M_{i,j,k} = 1 if the indices (i,j,k) correspond to the term A_{i,j} B_{j,k} in the sum for C_{i,k}.
        # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the term A_{i,j} B_{j,k} appears in C_{i,k} with coefficient 1.
        # So the tensor entry T_{i,j,k} is 1 if we are looking at the contribution to C_{i,k} from A_{i,j} and B_{j,k}.
        # This implies T_{i,j,k} = 1 for all i, j, k?
        # Let's check the decomposition:
        # C_{i,k} = sum_r W_{k,i,r} (sum_j U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k})
        # = sum_r W_{k,i,r} sum_j sum_{j'} U_{i,j,r} V_{j',k,r} A_{i,j} B_{j',k}
        # We want this to equal sum_j A_{i,j} B_{j,k}.
        # So we need sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = 1 for all i, j, k?
        # No, the indices j and j' must be the same for the term to be non-zero in the target.
        # The target is sum_j A_{i,j} B_{j,k}.
        # So the coefficient of A_{i,j} B_{j',k} in the LHS must be 1 if j=j' and 0 otherwise.
        # So we require: sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = delta_{j,j'}.
        # This is the condition.
        
        # Strassen terms:
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        # M2 = (A10 + A11)(B00) -> C10 += M2, C11 += M2
        # M3 = (A00)(B10 + B11) -> C00 += M3, C01 += M3
        # M4 = (A10)(B10) -> C10 += M4
        # M5 = (A01)(B01) -> C01 += M5
        # M6 = (A00 - A11)(B11 - B00) -> C01 += M6, C10 -= M6
        # M7 = (A11 - A10)(B01 - B00) -> C01 -= M7, C11 += M7
        
        # Let's use the standard Strassen factors:
        # U vectors (A coefficients):
        # 1: [1, 0, 0, 1]
        # 2: [0, 0, 1, 1]
        # 3: [1, 0, 0, 0]
        # 4: [0, 0, 1, 0]
        # 5: [0, 1, 0, 0]
        # 6: [1, 0, 0, -1]
        # 7: [0, 0, -1, 1]
        
        # V vectors (B coefficients):
        # 1: [1, 0, 0, 1]
        # 2: [1, 0, 0, 0]
        # 3: [0, 0, 1, 1]
        # 4: [0, 0, 1, 0]
        # 5: [0, 1, 0, 0]
        # 6: [-1, 0, 0, 1]
        # 7: [-1, 0, 1, 0]
        
        # W vectors (C coefficients, indexed k*n+i):
        # C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
        # Indices: 0:00, 1:01, 2:10, 3:11
        # 1: [1, 0, 0, 1] (C00, C11)
        # 2: [0, 0, 1, 1] (C10, C11)
        # 3: [1, 1, 0, 0] (C00, C01)
        # 4: [0, 0, 1, 0] (C10)
        # 5: [0, 1, 0, 0] (C01)
        # 6: [0, 1, -1, 0] (C01, -C10)
        # 7: [0, -1, 0, 1] (-C01, C11)
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]),
            ([0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]),
            ([0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
            ([1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0]),
            ([0, 0, -1, 1], [-1, 0, 1, 0], [0, -1, 0, 1])
        ]
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                # U: A is 2x2, indexed as [i*m+j] = [i*2+j]
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                # V: B is 2x10, indexed as [j*p+k] = [j*10+k]
                # For block q (columns 2q, 2q+1 of B):
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                # W: C is 2x10, indexed as [k*n+i] = [k*2+i]
                # For block q (columns 2q, 2q+1 of C):
                W[(2*q)*2 + 0, col] = c_vec[0]
                W[(2*q)*2 + 1, col] = c_vec[1]
                W[(2*q+1)*2 + 0, col] = c_vec[2]
                W[(2*q+1)*2 + 1, col] = c_vec[3]
                
                col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = 2, 2, 10
        R = U.shape[1]
        np.random.seed(42)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            # Compute C_computed using einsum for speed
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i*m+j]) * (sum_{j'} V[j'*p+k, r] B[j'*p+k])
            # Let's compute the intermediate sums
            # u_vals: (R, n*m) -> (R, 4)
            u_vals = np.dot(U, A_flat) # (R,)
            v_vals = np.dot(V, B_flat) # (R,)
            
            # C_computed[i,k] = sum_r W[k*n+i, r] u_vals[r] v_vals[r]
            # W is (20, R). We need to reshape W to (p, n, R) -> (10, 2, R)
            W_reshaped = W.reshape(p, n, R)
            # u_vals and v_vals are (R,)
            # Product term: (R,)
            prod = u_vals * v_vals
            
            # C_computed[i,k] = sum_r W_reshaped[k, i, r] prod[r]
            # C_computed = np.einsum('kir,r->ik', W_reshaped, prod)
            C_computed = np.einsum('kir,r->ik', W_reshaped, prod)
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    def check_rank_and_reduce(U, V, W, time_limit):
        """Check if the 35 tensors are linearly independent. If not, reduce."""
        R = U.shape[1]
        n, m, p = 2, 2, 10
        T_flat = np.zeros((R, n*m * m*p * p*n), dtype=float)
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            outer = np.einsum('a,b,c->abc', u, v, w)
            T_flat[r] = outer.flatten()
        
        # Check rank
        rank = np.linalg.matrix_rank(T_flat, tol=1e-10)
        if rank < R:
            # We have linear dependence.
            # We can find a basis and potentially reduce.
            # However, for Strassen, rank should be 35.
            # If it is less, we can try to remove columns.
            # For now, we just return the original if reduction is complex.
            pass
            
        return U, V, W

    # Build the rank-35 solution
    U, V, W = build_strassen_rank_35()
    
    # Verify correctness
    valid = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        R = 40
        U = np.zeros((4, R), dtype=float)
        V = np.zeros((20, R), dtype=float)
        W = np.zeros((20, R), dtype=float)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(10):
                    U[i*2+j, col] = 1
                    V[j*10+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        U, V, W = U, V, W
        valid = verify_decomposition(U, V, W)
        if not valid:
            raise ValueError("Construction failed")

    # Check for reduction (fast)
    if time.time() < deadline - 10:
        U, V, W = check_rank_and_reduce(U, V, W, deadline - 5)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
