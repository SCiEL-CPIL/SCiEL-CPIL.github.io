import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 5
    
    # Pre-allocate max possible rank (naive is 30, we aim for 19)
    max_rank = 30
    U = np.zeros((n * m, max_rank), dtype=object)
    V = np.zeros((m * p, max_rank), dtype=object)
    W = np.zeros((p * n, max_rank), dtype=object)
    col = 0

    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        if col >= max_rank:
            return
        for idx in u_indices:
            U[idx, col] += val
        for idx in v_indices:
            V[idx, col] += val
        for idx in w_indices:
            W[idx, col] += val
        col += 1

    # Laderman 2x2x2 decomposition
    # Indices:
    # A is 2x2 (i,j), B is 2x2 (j,k), C is 2x2 (i,k)
    # Tensor T[i,j,k]
    # Laderman terms (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # u_idx corresponds to (i,j) in 2x2 -> index i*2+j
    # v_idx corresponds to (j,k) in 2x2 -> index j*2+k
    # w_idx corresponds to (k,i) in 2x2 -> index k*2+i
    
    # Standard Laderman 7 terms:
    # 1. (a+d)(e+h) -> u: (0,0),(1,1); v: (0,0),(1,1); w: (0,0),(1,1)
    #    Wait, Laderman is usually given as products.
    #    Let's use the explicit rank-7 decomposition from literature.
    #    Terms:
    #    1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] -> (a+d)(e+h)
    #    2. u=[1,0,0,1], v=[0,1,1,0], w=[0,1,1,0] -> (a+d)(f+g) ... No, this is not Laderman.
    
    # Correct Laderman (rank 7):
    # m1 = (a+d)(e+h)
    # m2 = (a-b+2c+2d)(e+f+g+h) -- No, this is not right either.
    
    # Let's use the verified set from the previous attempt's "fallback" which was actually just 8 terms.
    # I need a valid 7-term set.
    # Reference: "Laderman's algorithm"
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 1. (a+d)(e+h)
    # 2. (a+d+2b)(e+h+2g)
    # 3. (a-b+2d)(e-f+2h)
    # 4. (c+d)(f+h)
    # 5. (a+c)(e+g)
    # 6. (b+d)(f+g)
    # 7. (a+c+2b)(e+g+2f)
    
    # Let's verify this set numerically in the code.
    
    def get_laderman_terms():
        # Returns list of (u_indices, v_indices, w_indices, u_vals, v_vals, w_vals)
        # u_indices: list of (i,j)
        # v_indices: list of (j,k)
        # w_indices: list of (k,i)
        
        terms = []
        
        # 1. (a+d)(e+h)
        # u: a(0,0), d(1,1) -> [0, 3]
        # v: e(0,0), h(1,1) -> [0, 3]
        # w: ? 
        # C = A B
        # C00 = ae + bg
        # C01 = af + bh
        # C10 = ce + dg
        # C11 = cf + dh
        # 
        # Let's just use a known valid rank-7 decomposition.
        # From "The rank of matrix multiplication" or similar.
        # A common one:
        # t1: (a+d)(e+h)
        # t2: (a-b+2c+2d)(e+f+g+h) -- This looks like it mixes too much.
        
        # Let's use the 7-term set from Strassen's 2x2x2? No, Strassen is 7 for 2x2x2? 
        # Yes, Strassen's algorithm is rank 7 for 2x2x2.
        # Strassen's 7 products:
        # M1 = (a+d)(e+h)
        # M2 = (a+d+2b)(e+h+2g)
        # M3 = (a-b+2d)(e-f+2h)
        # M4 = (c+d)(f+h)
        # M5 = (a+c)(e+g)
        # M6 = (b+d)(f+g)
        # M7 = (a+c+2b)(e+g+2f)
        
        # Let's check M1:
        # (a+d)(e+h) = ae + ah + de + dh
        # We want to form C.
        
        # Actually, there is a simpler representation using linear combinations.
        # I will define the terms as linear combinations of basis tensors.
        # Basis tensor E_{i,j,k} has 1 at (i,j,k) and 0 elsewhere.
        # The multiplication tensor T = sum_{i,j,k} E_{i,j,k} \otimes E_{j,k,l} \otimes E_{k,l,i} ? No.
        # T_{(i,j), (j,k), (k,i)} = 1.
        
        # I will use the explicit indices for Strassen/Laderman.
        # U vector for a term is a vector in R^{2x2} (flattened).
        # V vector in R^{2x2}.
        # W vector in R^{2x2}.
        
        # Term 1: (a+d)(e+h)
        # u = [1, 0, 0, 1] (a,d)
        # v = [1, 0, 0, 1] (e,h)
        # w = ? 
        # The product is (a+d)(e+h) = ae + ah + de + dh.
        # This contributes to C00 (ae), C01 (ah), C10 (de), C11 (dh).
        # So w must be [1, 1, 1, 1]?
        # If w=[1,1,1,1], then term is (a+d)(e+h)(1+1+1+1)? No, the tensor is sum u_i v_j w_k.
        # The contribution to C_{i,k} is sum_j U_{i,j} V_{j,k} W_{k,i}.
        # If we define the term by U, V, W vectors:
        # Contribution to C_{i,k} = sum_{j} U_{i,j} V_{j,k} W_{k,i} ?? No.
        # The tensor is T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
        # Here a=(i,j), b=(j,k), c=(k,i).
        # So for a fixed r, the contribution to the matrix C is:
        # C_{i,k} += sum_{j} U_{i*2+j, r} V_{j*2+k, r} W_{k*2+i, r}.
        
        # Let's construct the 7 Strassen terms explicitly.
        # 1. U: a,d (0,0; 1,1) -> [1,0,0,1]
        #    V: e,h (0,0; 1,1) -> [1,0,0,1]
        #    W: must satisfy sum_j U_{i,j} V_{j,k} W_{k,i} = ae + ah + de + dh for the parts it covers.
        #    Actually, Strassen's algorithm computes 7 products and then linear combinations.
        #    The tensor decomposition is a sum of 7 rank-1 tensors.
        #    Rank-1 tensor is U \otimes V \otimes W.
        #    So we need 7 triples (U, V, W) such that sum_{r=1}^7 U_r \otimes V_r \otimes W_r = T.
        
        # Let's use the known solution.
        # See "A rank-7 decomposition of the 2x2x2 matrix multiplication tensor".
        # One such set:
        # 1. U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1] -> (a+d)(e+h)(1+1+1+1) = 4(ae+ah+de+dh). Scaling?
        #    If we scale W by 1/4, we get ae+ah+de+dh.
        #    But we need integer/half-integer. 1/4 is not allowed.
        #    We can scale U, V, W.
        #    Let's use integers.
        
        # Let's try to find a valid set online or derive it.
        # Actually, the standard Strassen 7 products are:
        # M1 = (a+d)(e+h)
        # M2 = (a+d+2b)(e+h+2g)
        # M3 = (a-b+2d)(e-f+2h)
        # M4 = (c+d)(f+h)
        # M5 = (a+c)(e+g)
        # M6 = (b+d)(f+g)
        # M7 = (a+c+2b)(e+g+2f)
        
        # Let's check if these form a tensor decomposition.
        # C = M1 + M2 - M3 - M4 + M5 - M6 - M7 ? No, the linear combination is specific.
        # C00 = M1 - M2 + M3 + M4 - M5 - M6
        # C01 = M1 - M2 - M3 + M4 + M5 - M6
        # ...
        
        # This implies that the tensor T can be written as a sum of 7 rank-1 terms if we allow the "coefficients" to be absorbed into U, V, W.
        # Since the coefficients are +/- 1, we can just flip signs in U, V, or W.
        
        # Let's define the 7 terms based on the linear forms.
        # Term 1:
        # U1 = a+d -> [1, 0, 0, 1]
        # V1 = e+h -> [1, 0, 0, 1]
        # W1 = ?
        # The term contributes to all C_ik.
        # In the formula C00 = M1 ... M1 has coeff +1.
        # C01 = M1 ... M1 has coeff +1.
        # C10 = M1 ... M1 has coeff -1?
        # C11 = M1 ... M1 has coeff -1?
        
        # Let's look up the exact Strassen reconstruction:
        # C00 = M1 - M2 + M3 + M4 - M5 - M6
        # C01 = M1 - M2 - M3 + M4 + M5 - M6
        # C10 = M1 + M2 - M3 - M4 - M5 + M6
        # C11 = M1 + M2 + M3 - M4 + M5 + M6
        
        # Wait, where is M7?
        # Ah, the 7th product is often used to reduce rank further or is part of a different set.
        # Strassen's original 7 products are M1..M7.
        # The reconstruction uses M1..M6? No, it uses M1..M7.
        # Let's check a reliable source.
        # Wikipedia "Strassen algorithm":
        # C11 = M1 - M2 + M3 + M4 - M5 - M6
        # C12 = M1 - M2 - M3 + M4 + M5 - M6
        # C21 = M1 + M2 - M3 - M4 - M5 + M6
        # C22 = M1 + M2 + M3 - M4 + M5 + M6
        # This only uses M1..M6. What is M7?
        # M7 = (a+c+2b)(e+g+2f).
        # It seems M7 is not used in the reconstruction above?
        # No, that formula is for a specific variant.
        
        # Let's use a verified rank-7 decomposition from a paper.
        # "The tensor rank of 2x2x2 matrix multiplication is 7".
        # One decomposition:
        # 1. (1,0,0,1) \otimes (1,0,0,1) \otimes (1,1,1,1)
        # 2. (1,2,0,1) \otimes (1,0,2,1) \otimes (-1,-1,1,1)
        # 3. (1,-1,0,2) \otimes (1,-1,0,2) \otimes (-1,1,-1,1)
        # 4. (0,0,1,1) \otimes (0,2,1,1) \otimes (1,1,-1,-1)
        # 5. (1,0,1,0) \otimes (1,0,1,0) \otimes (-1,1,1,-1)
        # 6. (0,1,0,1) \otimes (2,1,0,1) \otimes (1,-1,1,1)
        # 7. (1,2,1,0) \otimes (1,0,2,0) \otimes (1,-1,-1,1)
        
        # I will implement this set and verify it numerically. If it fails, I will fallback to rank 8 (which is easier to construct) or rank 30.
        
        laderman_7 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
            ([1, 2, 0, 1], [1, 0, 2, 1], [-1, -1, 1, 1]),
            ([1, -1, 0, 2], [1, -1, 0, 2], [-1, 1, -1, 1]),
            ([0, 0, 1, 1], [0, 2, 1, 1], [1, 1, -1, -1]),
            ([1, 0, 1, 0], [1, 0, 1, 0], [-1, 1, 1, -1]),
            ([0, 1, 0, 1], [2, 1, 0, 1], [1, -1, 1, 1]),
            ([1, 2, 1, 0], [1, 0, 2, 0], [1, -1, -1, 1]),
        ]
        
        # Verify locally
        def verify_laderman(terms):
            T = np.zeros((2, 2, 2), dtype=object)
            for u, v, w in terms:
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            # u is flattened (i,j) -> i*2+j
                            # v is flattened (j,k) -> j*2+k
                            # w is flattened (k,i) -> k*2+i
                            T[i, j, k] += u[i*2+j] * v[j*2+k] * w[k*2+i]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if T[i, j, k] != 1:
                            return False
            return True
            
        if not verify_laderman(laderman_7):
            # Fallback to rank 8 if 7 fails
            # Rank 8 is easy: just use the 8 terms from the naive 2x2x2?
            # No, naive 2x2x2 is rank 4? No, naive is sum of 4 terms?
            # T[i,j,k] = 1.
            # Term (i,j,k): U[i*2+j]=1, V[j*2+k]=1, W[k*2+i]=1.
            # There are 8 such terms (i,j,k in 0..1).
            # So rank 8 is always valid.
            laderman_7 = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = [0]*4; u[i*2+j] = 1
                        v = [0]*4; v[j*2+k] = 1
                        w = [0]*4; w[k*2+i] = 1
                        laderman_7.append((u, v, w))
            
        return laderman_7

    terms = get_laderman_terms()
    
    # Part A: j in {0,1}, k in {0,1}
    for u, v, w in terms:
        # Map local 2x2 indices to global
        # u_idx: i*3 + j (local j is 0,1)
        # v_idx: j*5 + k (local k is 0,1)
        # w_idx: k*2 + i
        
        u_g = [0]*6
        v_g = [0]*15
        w_g = [0]*10
        
        for li in range(2):
            for lj in range(2):
                u_g[li*3 + lj] = u[li*2 + lj]
                
        for lj in range(2):
            for lk in range(2):
                v_g[lj*5 + lk] = v[lj*2 + lk]
                
        for lk in range(2):
            for li in range(2):
                w_g[lk*2 + li] = w[lk*2 + li]
                
        add_term(u_g, v_g, w_g, 1)

    # Part B: j in {0,1}, k in {2,3}
    for u, v, w in terms:
        u_g = [0]*6
        v_g = [0]*15
        w_g = [0]*10
        
        for li in range(2):
            for lj in range(2):
                u_g[li*3 + lj] = u[li*2 + lj]
                
        for lj in range(2):
            for lk in range(2):
                # Map local k (0,1) to global k (2,3)
                v_g[lj*5 + (lk+2)] = v[lj*2 + lk]
                
        for lk in range(2):
            for li in range(2):
                # Map local k (0,1) to global k (2,3)
                w_g[(lk+2)*2 + li] = w[lk*2 + li]
                
        add_term(u_g, v_g, w_g, 1)

    # Part C: j in {0,1}, k=4
    # 4 terms
    for li in range(2):
        for lj in range(2):
            u_g = [0]*6
            v_g = [0]*15
            w_g = [0]*10
            
            u_g[li*3 + lj] = 1
            v_g[lj*5 + 4] = 1
            w_g[4*2 + li] = 1
            
            add_term(u_g, v_g, w_g, 1)

    # Part D: j=2
    # 1 term
    u_g = [0]*6
    v_g = [0]*15
    w_g = [0]*10
    
    for li in range(2):
        u_g[li*3 + 2] = 1
        
    for lk in range(5):
        v_g[2*5 + lk] = 1
        
    for lk in range(5):
        for li in range(2):
            w_g[lk*2 + li] = 1
            
    add_term(u_g, v_g, w_g, 1)

    R = col
    U = U[:, :R]
    V = V[:, :R]
    W = W[:, :R]

    # Final Verification
    def verify(U, V, W):
        T = np.zeros((n * m, m * p, p * n), dtype=object)
        for r in range(U.shape[1]):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i * m + j, j * p + k, k * n + i] != 1:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    jj, k = divmod(b, p)
                    kk, ii = divmod(c, n)
                    if not (i == ii and j == jj and k == kk):
                        if T[a, b, c] != 0:
                            return False
        return True

    if not verify(U, V, W):
        # Fallback to naive
        R = n * m * p
        U = np.zeros((n * m, R), dtype=object)
        V = np.zeros((m * p, R), dtype=object)
        W = np.zeros((p * n, R), dtype=object)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        R = c

    def to_list(mat):
        return [[int(mat[i, j]) if mat[i, j] != 0 else 0 for j in range(mat.shape[1])]
                for i in range(mat.shape[0])]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
