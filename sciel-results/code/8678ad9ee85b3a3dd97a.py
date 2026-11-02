import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 35s margin before 150s kill

    def build_tensor(n, m, p):
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def verify(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_true = build_tensor(n, m, p)
        return np.max(np.abs(T_recon - T_true)) < tol

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2_block(j0, j1, k0, k1):
        """Return 7 rank-1 terms for 2x2 * 2x2 matrix multiplication."""
        # Standard Strassen terms for C = A*B where A, B are 2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        terms = []
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For our indexing:
        # U indices: a00=i0*m+j0, a01=i0*m+j1, a10=i1*m+j0, a11=i1*m+j1
        # V indices: b00=j0*p+k0, b01=j0*p+k1, b10=j1*p+k0, b11=j1*p+k1
        # W indices: c00=k0*n+0, c01=k1*n+0, c10=k0*n+1, c11=k1*n+1
        
        # Each term is (U_coeffs, V_coeffs, W_coeffs) where coeffs are dicts
        # mapping index -> value
        
        # M1: (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        u = {0*m+j0: 1, 1*m+j1: 1}
        v = {j0*p+k0: 1, j1*p+k1: 1}
        w = {k0*n+0: 1, k1*n+1: 1}
        terms.append((u, v, w))
        
        # M2: (a10+a11)b00 -> C10 += M2, C11 -= M2
        u = {1*m+j0: 1, 1*m+j1: 1}
        v = {j0*p+k0: 1}
        w = {k0*n+1: 1, k1*n+1: -1}
        terms.append((u, v, w))
        
        # M3: a00(b10-b11) -> C01 += M3, C11 += M3
        u = {0*m+j0: 1}
        v = {j1*p+k0: 1, j1*p+k1: -1}
        w = {k1*n+0: 1, k1*n+1: 1}
        terms.append((u, v, w))
        
        # M4: a11(b01-b00) -> C00 += M4, C10 += M4
        u = {1*m+j1: 1}
        v = {j0*p+k1: 1, j0*p+k0: -1}
        w = {k0*n+0: 1, k0*n+1: 1}
        terms.append((u, v, w))
        
        # M5: (a00+a01)b11 -> C00 -= M5, C01 += M5
        u = {0*m+j0: 1, 0*m+j1: 1}
        v = {j1*p+k1: 1}
        w = {k0*n+0: -1, k1*n+0: 1}
        terms.append((u, v, w))
        
        # M6: (a10-a00)(b00+b10) -> C11 += M6
        u = {1*m+j0: 1, 0*m+j0: -1}
        v = {j0*p+k0: 1, j1*p+k0: 1}
        w = {k1*n+1: 1}
        terms.append((u, v, w))
        
        # M7: (a01-a11)(b01+b11) -> C00 += M7
        u = {0*m+j1: 1, 1*m+j1: -1}
        v = {j0*p+k1: 1, j1*p+k1: 1}
        w = {k0*n+0: 1}
        terms.append((u, v, w))
        
        return terms

    def build_hybrid_decomposition(n, m, p):
        """Build the hybrid Strassen+schoolbook decomposition."""
        U_rows = []
        V_rows = []
        W_rows = []

        def add_term(u_dict, v_dict, w_dict):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            for idx, val in u_dict.items():
                U_vec[idx] = val
            for idx, val in v_dict.items():
                V_vec[idx] = val
            for idx, val in w_dict.items():
                W_vec[idx] = val
            U_rows.append(U_vec)
            V_rows.append(V_vec)
            W_rows.append(W_vec)

        # Strassen blocks for (j=0,1) with k pairs (0,1), (2,3), (4,5)
        for k0 in [0, 2, 4]:
            k1 = k0 + 1
            terms = strassen_2x2_block(0, 1, k0, k1)
            for u, v, w in terms:
                add_term(u, v, w)

        # Schoolbook for A[:,0:2] @ B[0:2, 6]
        for i in range(n):
            for j in [0, 1]:
                add_term({i*m+j: 1}, {j*p+6: 1}, {6*n+i: 1})

        # Schoolbook for A[:,2] @ B[2,:]
        for i in range(n):
            for k in range(p):
                add_term({i*m+2: 1}, {2*p+k: 1}, {k*n+i: 1})

        U = np.column_stack(U_rows)
        V = np.column_stack(V_rows)
        W = np.column_stack(W_rows)
        return U, V, W

    def try_reduce_rank(U, V, W, target_rank, time_budget=20.0):
        """Try to reduce the rank of the decomposition to target_rank."""
        start = time.time()
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        # Try to find linear dependencies by random linear combinations
        # We'll try to merge pairs of columns that are "close" in some sense
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        for attempt in range(50):
            if time.time() - start > time_budget:
                break
            if best_rank <= target_rank:
                break
            
            # Pick two random columns to try to merge
            c1 = random.randint(0, current_rank - 1)
            c2 = random.randint(0, current_rank - 1)
            if c1 == c2:
                continue
            
            # Try a random linear combination: new_col = alpha*col1 + beta*col2
            alpha = random.uniform(-2, 2)
            beta = random.uniform(-2, 2)
            
            if abs(alpha) < 1e-10 or abs(beta) < 1e-10:
                continue
            
            # Create a trial decomposition
            trial_U = current_U.copy()
            trial_V = current_V.copy()
            trial_W = current_W.copy()
            
            # Replace column c2 with the linear combination
            trial_U[:, c2] = alpha * current_U[:, c1] + beta * current_V[:, c1] * 0 + beta * current_U[:, c2]
            # Wait, this isn't right. The tensor is sum_r U[:,r]*V[:,r]*W[:,r]
            # If we replace column r with a combination, we need to be careful.
            # Actually, we can't just combine columns arbitrarily because the tensor
            # is a sum of rank-1 terms. We need to find a different approach.
            
            # Alternative: try to find if two columns are linearly dependent
            # in the tensor space. This is complex.
            # Instead, let's try a different strategy: use the structure of the
            # Strassen decomposition to find actual dependencies.
            
            break  # Skip this approach for now
        
        return best_U, best_V, best_W

    def polish_decomposition(U, V, W, n, m, p, time_budget=10.0):
        """Try to polish the decomposition for better numerical stability."""
        start = time.time()
        # Convert to Fraction for exact arithmetic if needed
        # For now, just verify and return
        if verify(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
        return U, V, W

    # Build the hybrid decomposition
    U, V, W = build_hybrid_decomposition(n, m, p)
    
    # Verify
    if verify(U, V, W, n, m, p, tol=1e-9):
        rank = U.shape[1]
        
        # Try to reduce rank if possible
        if time.time() < deadline - 30:
            # Try various reduction strategies
            for target_rank in range(30, rank):
                if time.time() > deadline - 15:
                    break
                reduced_U, reduced_V, reduced_W = try_reduce_rank(U, V, W, target_rank, time_budget=10.0)
                if verify(reduced_U, reduced_V, reduced_W, n, m, p, tol=1e-9):
                    U, V, W = reduced_U, reduced_V, reduced_W
                    rank = U.shape[1]
                    break
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
