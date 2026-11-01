"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the full tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Verify a decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
    def verify(U, V, W, tol=1e-6):
        rank = U.shape[1]
        T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_decomp, T, atol=tol)

    # Schoolbook decomposition of rank n*m*p = 32 (guaranteed valid baseline).
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    # Strassen-based decomposition: M_{2,2,8} = M_{2,2,2} ⊗ I_4
    # Rank = 7 * 4 = 28
    def strassen_decomposition():
        # Strassen's 7 multiplications for 2x2 matrix multiplication
        # S1 = (A11 + A22) * (B11 + B22)
        # S2 = (A21 + A22) * B11
        # S3 = A11 * (B12 - B22)
        # S4 = A22 * (B21 - B11)
        # S5 = (A11 + A12) * B22
        # S6 = (A21 - A11) * (B11 + B12)
        # S7 = (A12 - A22) * (B21 + B22)
        
        # C11 = S1 + S4 - S5 + S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S2 + S3 + S6
        
        # For M_{2,2,8}, we have 4 independent 2x2 blocks (since p=8=2*4)
        # Each block uses 7 Strassen multiplications
        # Total rank = 7 * 4 = 28
        
        rank = 28
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        # Strassen coefficients for A matrix (input)
        # A indices: a = i*m + j where i is row, j is col
        # For 2x2: A11=a0, A12=a1, A21=a2, A22=a3
        
        # Strassen coefficients for B matrix (input)
        # B indices: b = j*p + k where j is row, k is col
        # For 2x2 block: B11, B12, B21, B22
        
        # Strassen coefficients for C matrix (output)
        # C indices: c = k*n + i where k is col, i is row
        # For 2x2 block: C11, C12, C21, C22
        
        # The 7 Strassen terms for a single 2x2x2 multiplication:
        # Term 1: (A11+A22)*(B11+B22) -> contributes to C11, C22
        # Term 2: (A21+A22)*B11 -> contributes to C21
        # Term 3: A11*(B12-B22) -> contributes to C12
        # Term 4: A22*(B21-B11) -> contributes to C21
        # Term 5: (A11+A12)*B22 -> contributes to C11, C12
        # Term 6: (A21-A11)*(B11+B12) -> contributes to C22
        # Term 7: (A12-A22)*(B21+B22) -> contributes to C11, C22
        
        # For each of the 4 blocks (k=0,1,2,3 corresponding to columns 4k,4k+1,4k+2,4k+3 of B and C):
        # Wait, let me reconsider the structure.
        
        # M_{2,2,8}: A is 2x2, B is 2x8, C is 2x8
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # We can view this as 4 independent 2x2 multiplications where:
        # Block t (t=0,1,2,3): B[:, 2t:2t+2] is 2x2, C[:, 2t:2t+2] is 2x2
        # But wait, that's not quite right because the Strassen algorithm works on 2x2 blocks.
        
        # Actually, M_{2,2,8} can be seen as: for each output column k, C[:,k] = A * B[:,k]
        # This is 8 independent 2x1 multiplications, but we can group them.
        
        # Better view: M_{2,2,8} = M_{2,2,2} ⊗ I_4 where I_4 is 4x4 identity
        # This means we have 4 independent copies of M_{2,2,2}
        
        # For block t (t=0,1,2,3):
        # Input A is the same 2x2 matrix (shared across all blocks)
        # Input B_t = B[:, 2t:2t+2] is 2x2
        # Output C_t = C[:, 2t:2t+2] is 2x2
        
        # Wait, that's not right either. Let me re-examine the tensor structure.
        
        # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
        # With n=2, m=2, p=8:
        # a = 2i+j, b = 2j+k, c = 2k+i
        # i,j in {0,1}, k in {0,...,7}
        
        # So for fixed i,j,k: T[2i+j, 2j+k, 2k+i] = 1
        
        # This is NOT a simple Kronecker product structure. Let me reconsider.
        
        # Actually, M_{2,2,8} means: (2x2) * (2x8) = (2x8)
        # The tensor has indices (i,j,k) where i=row of A, j=col of A/row of B, k=col of B/col of C
        # T[i,j,k] = delta_{j, row_of_B} ... 
        
        # Let me use the standard matrix multiplication tensor definition:
        # For C = AB where A is n x m, B is m x p, C is n x p
        # T[i,j,k,l] = delta_{j,k} where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
        # In 3-way form: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
        # T[a,b,c] = 1 iff the indices are consistent
        
        # With n=2, m=2, p=8:
        # a = i*2 + j (i in {0,1}, j in {0,1})
        # b = j*8 + k (j in {0,1}, k in {0,...,7})
        # c = k*2 + i (k in {0,...,7}, i in {0,1})
        
        # T[a,b,c] = 1 iff there exist i,j,k such that:
        # a = 2i+j, b = 8j+k, c = 2k+i
        
        # This is equivalent to: for each (i,j,k), T[2i+j, 8j+k, 2k+i] = 1
        
        # Now, can we decompose this using Strassen?
        # Strassen works for M_{2,2,2}. For M_{2,2,8}, we can think of it as:
        # C[:,k] = A * B[:,k] for k=0,...,7
        # This is 8 separate 2x2 * 2x1 = 2x1 multiplications
        # Or equivalently, 4 separate 2x2 * 2x2 = 2x2 multiplications if we group columns
        
        # Let's group: for t=0,1,2,3, let k = 2t, 2t+1
        # Then C[:, 2t:2t+2] = A * B[:, 2t:2t+2]
        # This is 4 independent 2x2 * 2x2 = 2x2 multiplications
        
        # Each 2x2 * 2x2 multiplication can be done with Strassen rank 7
        # Total rank = 4 * 7 = 28
        
        # Now let's construct the decomposition:
        # For block t (t=0,1,2,3):
        # B_t = B[:, 2t:2t+2] has entries B[j, 2t+s] for j in {0,1}, s in {0,1}
        # C_t = C[:, 2t:2t+2] has entries C[i, 2t+s] for i in {0,1}, s in {0,1}
        
        # Strassen for block t:
        # S1_t = (A11+A22) * (B_t[0,0] + B_t[1,1])
        # S2_t = (A21+A22) * B_t[0,0]
        # S3_t = A11 * (B_t[0,1] - B_t[1,1])
        # S4_t = A22 * (B_t[1,0] - B_t[0,0])
        # S5_t = (A11+A12) * B_t[1,1]
        # S6_t = (A21-A11) * (B_t[0,0] + B_t[0,1])
        # S7_t = (A12-A22) * (B_t[1,0] + B_t[1,1])
        
        # C_t[0,0] = S1_t + S4_t - S5_t + S7_t
        # C_t[0,1] = S3_t + S5_t
        # C_t[1,0] = S2_t + S4_t
        # C_t[1,1] = S1_t - S2_t + S3_t + S6_t
        
        # In tensor terms, each Strassen term is a rank-1 component.
        # For block t, term r (r=0,...,6):
        # U vector: coefficients of A
        # V vector: coefficients of B_t
        # W vector: coefficients of C_t
        
        # Let's define the 7 Strassen terms for 2x2*2x2:
        # Term 0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1]  (S1: (A11+A22)(B11+B22) -> C11+=1, C22+=-1)
        # Wait, I need to be careful with the output coefficients.
        
        # Let me redefine: for each Strassen term, we have:
        # U = coefficients for A (4 entries: A11, A12, A21, A22)
        # V = coefficients for B (4 entries: B11, B12, B21, B22)
        # W = coefficients for C (4 entries: C11, C12, C21, C22)
        
        # Term 0 (S1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        #   (A11+A22)*(B11+B22) contributes +1 to C11 and +1 to C22
        #   But C22 = S1 - S2 + S3 + S6, so S1 contributes +1 to C22
        #   C11 = S1 + S4 - S5 + S7, so S1 contributes +1 to C11
        #   So W=[1,0,0,1]
        
        # Term 1 (S2): U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
        #   (A21+A22)*B11 contributes +1 to C21
        #   C21 = S2 + S4, so S2 contributes +1 to C21
        #   W=[0,0,1,0]
        
        # Term 2 (S3): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
        #   A11*(B12-B22) contributes +1 to C12 and +1 to C22
        #   C12 = S3 + S5, so S3 contributes +1 to C12
        #   C22 = S1 - S2 + S3 + S6, so S3 contributes +1 to C22
        #   W=[0,1,0,1]
        
        # Term 3 (S4): U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]
        #   A22*(B21-B11) contributes +1 to C11 and +1 to C21
        #   C11 = S1 + S4 - S5 + S7, so S4 contributes +1 to C11
        #   C21 = S2 + S4, so S4 contributes +1 to C21
        #   W=[1,0,1,0]
        
        # Term 4 (S5): U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
        #   (A11+A12)*B22 contributes -1 to C11 and +1 to C12
        #   C11 = S1 + S4 - S5 + S7, so S5 contributes -1 to C11
        #   C12 = S3 + S5, so S5 contributes +1 to C12
        #   W=[-1,1,0,0]
        
        # Term 5 (S6): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
        #   (A21-A11)*(B11+B12) contributes +1 to C22
        #   C22 = S1 - S2 + S3 + S6, so S6 contributes +1 to C22
        #   W=[0,0,0,1]
        
        # Term 6 (S7): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1]
        #   (A12-A22)*(B21+B22) contributes +1 to C11 and +1 to C22
        #   C11 = S1 + S4 - S5 + S7, so S7 contributes +1 to C11
        #   C22 = S1 - S2 + S3 + S6... wait, S7 is not in C22 formula
        #   Let me recheck: C22 = S1 - S2 + S3 + S6
        #   So S7 does NOT contribute to C22. Let me recheck Strassen.
        
        # Standard Strassen:
        # C11 = S1 + S4 - S5 + S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S2 + S3 + S6
        
        # So S7 only contributes to C11.
        # Term 6 (S7): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
        #   (A12-A22)*(B21+B22) contributes +1 to C11
        #   W=[1,0,0,0]
        
        # Let me verify: sum of all W contributions:
        # C11: S1(+1) + S4(+1) + S5(-1) + S7(+1) = S1+S4-S5+S7 ✓
        # C12: S3(+1) + S5(+1) = S3+S5 ✓
        # C21: S2(+1) + S4(+1) = S2+S4 ✓
        # C22: S1(+1) + S2(-1) + S3(+1) + S6(+1) = S1-S2+S3+S6 ✓
        
        # Great! Now let's construct the full decomposition for M_{2,2,8}.
        
        # For block t (t=0,1,2,3):
        # B_t[j,s] = B[j, 2t+s] for j in {0,1}, s in {0,1}
        # In the full B matrix (2x8), index b = j*8 + k where k = 2t+s
        # So B_t[0,0] = B[0, 2t], B_t[0,1] = B[0, 2t+1], B_t[1,0] = B[1, 2t], B_t[1,1] = B[1, 2t+1]
        
        # C_t[i,s] = C[i, 2t+s]
        # In the full C matrix (2x8), index c = k*2 + i where k = 2t+s
        # So C_t[0,0] = C[0, 2t], C_t[0,1] = C[0, 2t+1], C_t[1,0] = C[1, 2t], C_t[1,1] = C[1, 2t+1]
        # In terms of c index: C[0, 2t] has c = (2t)*2 + 0 = 4t, C[0, 2t+1] has c = (2t+1)*2 + 0 = 4t+2
        # C[1, 2t] has c = (2t)*2 + 1 = 4t+1, C[1, 2t+1] has c = (2t+1)*2 + 1 = 4t+3
        
        # So for block t:
        # C_t[0,0] -> c = 4t
        # C_t[0,1] -> c = 4t+2
        # C_t[1,0] -> c = 4t+1
        # C_t[1,1] -> c = 4t+3
        
        # Now let's build U, V, W for all 28 terms.
        
        # Strassen terms (U, V, W) for 2x2*2x2:
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: [A11, A12, A21, A22], V: [B11, B12, B21, B22], W: [C11, C12, C21, C22]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # S1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # S2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # S3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # S4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # S5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # S6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # S7
        ]
        
        c_idx = 0
        for t in range(4):
            for r in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[r]
                
                # U: A is shared across all blocks, so same for each t
                # A[i,j] has index a = i*2 + j
                # A11 = a=0, A12 = a=1, A21 = a=2, A22 = a=3
                for ai, coeff in enumerate(U_coeffs):
                    if coeff != 0:
                        U[ai, c_idx] = coeff
                
                # V: B_t[j,s] = B[j, 2t+s]
                # B[j,k] has index b = j*8 + k
                # B_t[0,0] = B[0, 2t] -> b = 0*8 + 2t = 2t
                # B_t[0,1] = B[0, 2t+1] -> b = 0*8 + 2t+1 = 2t+1
                # B_t[1,0] = B[1, 2t] -> b = 1*8 + 2t = 8+2t
                # B_t[1,1] = B[1, 2t+1] -> b = 1*8 + 2t+1 = 8+2t+1
                b_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for bi, coeff in enumerate(V_coeffs):
                    if coeff != 0:
                        V[b_indices[bi], c_idx] = coeff
                
                # W: C_t[i,s] = C[i, 2t+s]
                # C[i,k] has index c = k*2 + i
                # C_t[0,0] = C[0, 2t] -> c = 2t*2 + 0 = 4t
                # C_t[0,1] = C[0, 2t+1] -> c = (2t+1)*2 + 0 = 4t+2
                # C_t[1,0] = C[1, 2t] -> c = 2t*2 + 1 = 4t+1
                # C_t[1,1] = C[1, 2t+1] -> c = (2t+1)*2 + 1 = 4t+3
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for wi, coeff in enumerate(W_coeffs):
                    if coeff != 0:
                        W[w_indices[wi], c_idx] = coeff
                
                c_idx += 1
        
        return U, V, W

    # Try Strassen decomposition (rank 28)
    U_str, V_str, W_str = strassen_decomposition()
    if verify(U_str, V_str, W_str, tol=1e-6):
        best_U, best_V, best_W = U_str, V_str, W_str
        best_rank = 28
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = 32

    # Try to find even lower rank using optimization
    # Target ranks: 24, 25, 26, 27
    for target_rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        rank = target_rank
        # Fix W to a random matrix with orthonormal columns
        if rank <= pn:
            Q, _ = np.linalg.qr(np.random.randn(pn, pn))
            W_fixed = Q[:, :rank]
        else:
            W_fixed = np.random.randn(pn, rank)
            norms = np.linalg.norm(W_fixed, axis=0, keepdims=True)
            norms[norms == 0] = 1
            W_fixed /= norms

        found = False
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(nm, rank)
            V_init = np.random.randn(mp, rank)
            x_init = np.concatenate([U_init.flatten(), V_init.flatten()])
            
            def objective(x, U_init=U_init, V_init=V_init, W_fixed=W_fixed, rank=rank):
                if time.time() > deadline:
                    return 1e10
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W_fixed)
                diff = T_decomp - T
                return np.sum(diff * diff)
            
            # Simple gradient descent
            x = x_init.copy()
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                f = objective(x)
                if f < 1e-6:
                    break
                # Numerical gradient
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    if time.time() > deadline:
                        break
                    x[i] += 1e-4
                    f_plus = objective(x)
                    x[i] -= 2e-4
                    f_minus = objective(x)
                    x[i] += 1e-4
                    grad[i] = (f_plus - f_minus) / 2e-4
                x -= lr * grad
                # Decay learning rate
                lr *= 0.95
            
            U_opt = x[:nm*rank].reshape(nm, rank)
            V_opt = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            
            if verify(U_opt, V_opt, W_fixed, tol=1e-5):
                # Try rounding to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_fixed * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, tol=1e-5):
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
