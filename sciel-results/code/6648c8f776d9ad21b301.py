import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Start with schoolbook
    rank = n * m * p  # 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Try to apply Strassen to 2x2x2 blocks
    # For block (i0, i1, j0, j1, k0, k1):
    # Replace 8 schoolbook terms with 7 Strassen terms
    
    def apply_strassen(U, V, W, i0, i1, j0, j1, k0, k1, rank):
        """Replace 8 schoolbook terms for 2x2x2 block with 7 Strassen terms.
        Returns new rank (reduced by 1)."""
        # Find the 8 schoolbook columns
        cols = []
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    # Find column where U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
                    for c in range(rank):
                        if (U[i*m+j, c] == 1 and V[j*p+k, c] == 1 and W[k*n+i, c] == 1):
                            cols.append(c)
                            break
        if len(cols) != 8:
            return U, V, W, rank  # Can't find all 8
        
        # Create new U, V, W with 1 fewer column
        new_rank = rank - 1
        new_U = np.zeros((n*m, new_rank))
        new_V = np.zeros((m*p, new_rank))
        new_W = np.zeros((p*n, new_rank))
        
        # Copy non-Strassen columns
        col_map = []
        new_col = 0
        for c in range(rank):
            if c not in cols:
                new_U[:, new_col] = U[:, c]
                new_V[:, new_col] = V[:, c]
                new_W[:, new_col] = W[:, c]
                col_map.append(new_col)
                new_col += 1
        
        # Add 7 Strassen columns
        # M1 = (a+d)(e+h) where a=A[i0,j0], d=A[i1,j1], e=B[j0,k0], h=B[j1,k1]
        # M2 = (c+d)e where c=A[i1,j0], e=B[j0,k0]
        # M3 = a(f-h) where f=B[j0,k1], h=B[j1,k1]
        # M4 = d(g-e) where g=B[j1,k0]
        # M5 = (a+b)h where b=A[i0,j1]
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        strassen_terms = [
            # (U coeffs, V coeffs, W coeffs)
            # M1
            ({(i0,j0): 1, (i1,j1): 1}, {(j0,k0): 1, (j1,k1): 1}, 
             {(k0,i0): 1, (k1,i1): 1}),
            # M2
            ({(i1,j0): 1, (i1,j1): 1}, {(j0,k0): 1}, 
             {(k0,i1): 1}),
            # M3
            ({(i0,j0): 1}, {(j0,k1): 1, (j1,k1): -1}, 
             {(k1,i0): 1}),
            # M4
            ({(i1,j1): 1}, {(j1,k0): 1, (j0,k0): -1}, 
             {(k0,i0): 1, (k0,i1): 1}),
            # M5
            ({(i0,j0): 1, (i0,j1): 1}, {(j1,k1): 1}, 
             {(k0,i0): -1, (k1,i0): 1}),
            # M6
            ({(i1,j0): 1, (i0,j0): -1}, {(j0,k0): 1, (j0,k1): 1}, 
             {(k1,i1): 1}),
            # M7
            ({(i0,j1): 1, (i1,j1): -1}, {(j1,k0): 1, (j1,k1): 1}, 
             {(k0,i0): 1}),
        ]
        
        for term_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            c = new_col + term_idx
            for (i, j), coeff in u_coeffs.items():
                new_U[i*m + j, c] += coeff
            for (j, k), coeff in v_coeffs.items():
                new_V[j*p + k, c] += coeff
            for (k, i), coeff in w_coeffs.items():
                new_W[k*n + i, c] += coeff
        
        return new_U, new_V, new_W, new_rank
    
    # Verify function
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    # Try applying Strassen to various 2x2x2 blocks
    # We need to be careful about overlapping blocks
    
    # First try: block (0,1, 0,1, 0,1)
    U2, V2, W2, rank2 = apply_strassen(U, V, W, 0, 1, 0, 1, 0, 1, rank)
    if verify(U2, V2, W2, rank2):
        U, V, W, rank = U2, V2, W2, rank2
    
    # Second try: block (0,1, 0,1, 2,3)
    U3, V3, W3, rank3 = apply_strassen(U, V, W, 0, 1, 0, 1, 2, 3, rank)
    if verify(U3, V3, W3, rank3):
        U, V, W, rank = U3, V3, W3, rank3
    
    # Third try: block (0,1, 0,1, 4,5)
    U4, V4, W4, rank4 = apply_strassen(U, V, W, 0, 1, 0, 1, 4, 5, rank)
    if verify(U4, V4, W4, rank4):
        U, V, W, rank = U4, V4, W4, rank4
    
    # Try more blocks
    # Block (1,2, 0,1, 0,1)
    U5, V5, W5, rank5 = apply_strassen(U, V, W, 1, 2, 0, 1, 0, 1, rank)
    if verify(U5, V5, W5, rank5):
        U, V, W, rank = U5, V5, W5, rank5
    
    # Block (0,1, 1,2, 0,1)
    U6, V6, W6, rank6 = apply_strassen(U, V, W, 0, 1, 1, 2, 0, 1, rank)
    if verify(U6, V6, W6, rank6):
        U, V, W, rank = U6, V6, W6, rank6
    
    # Try all possible 2x2x2 blocks
    for i0 in range(n-1):
        i1 = i0 + 1
        for j0 in range(m-1):
            j1 = j0 + 1
            for k0 in range(p-1):
                k1 = k0 + 1
                if time.time() > deadline - 10:
                    break
                U_t, V_t, W_t, rank_t = apply_strassen(U, V, W, i0, i1, j0, j1, k0, k1, rank)
                if verify(U_t, V_t, W_t, rank_t):
                    U, V, W, rank = U_t, V_t, W_t, rank_t
                if time.time() > deadline - 10:
                    break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
