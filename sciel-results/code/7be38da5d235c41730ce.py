"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Laderman's rank-23 decomposition for 3x3 x 3x3
    # We split the 3x6 matrix into two 3x3 blocks: B = [B1 | B2]
    # C = A * B = A * B1 + A * B2
    # Apply Laderman to each: rank 23 + 23 = 46
    
    # Laderman's 23 terms for C = A * B (3x3 * 3x3)
    # A indices: 0..8 (row-major: a,b,c,d,e,f,g,h,i)
    # B indices: 0..8 (row-major: j,k,l,m,n,o,p,q,r)
    # C indices: 0..8 (row-major: C00,C01,C02,C10,C11,C12,C20,C21,C22)
    
    # Each term: (alpha[9], beta[9], gamma[9])
    # M_t = (sum_a alpha[a]*A[a]) * (sum_b beta[b]*B[b])
    # C[c] += gamma[c] * M_t
    
    laderman_terms = []
    
    # M1 = a * (j - m + p + q)
    laderman_terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,-1,0,0,1,1,0], [1,0,0,0,0,0,0,0,1]))
    # M2 = (a+b) * m
    laderman_terms.append(([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # M3 = (a+c) * p
    laderman_terms.append(([1,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0]))
    # M4 = i * (j + k - m - n)
    laderman_terms.append(([0,0,0,0,0,0,0,0,1], [1,1,0,-1,-1,0,0,0,0], [1,0,0,0,0,0,0,0,1]))
    # M5 = (g+i) * k
    laderman_terms.append(([0,0,0,0,0,0,1,0,1], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # M6 = (g+i) * n
    laderman_terms.append(([0,0,0,0,0,0,1,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # M7 = (c+i) * (m + n - p - q)
    laderman_terms.append(([0,1,0,0,0,0,0,0,1], [0,0,0,1,1,0,-1,-1,0], [0,1,0,0,0,0,0,1,0]))
    # M8 = (a+c+g+i-e) * (j+r)
    laderman_terms.append(([1,0,1,0,-1,0,1,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,-1]))
    # M9 = (c+i-e) * (j+r)
    laderman_terms.append(([0,0,1,0,-1,0,0,0,1], [1,0,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0,0]))
    # M10 = (a+g) * (l - n - q)
    laderman_terms.append(([1,0,0,0,0,0,1,0,0], [0,0,1,0,-1,0,0,-1,0], [0,0,0,1,0,0,0,0,0]))
    # M11 = (a+b+g+i) * (l - o)
    laderman_terms.append(([1,1,0,0,0,0,1,0,1], [0,0,1,0,0,-1,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # M12 = (b+c+e) * (n - o)
    laderman_terms.append(([0,1,1,0,1,0,0,0,0], [0,0,0,0,1,-1,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # M13 = (b+c+g+i) * (m - p)
    laderman_terms.append(([0,1,1,0,0,0,1,0,1], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]))
    # M14 = (a+c+e) * (k - l)
    laderman_terms.append(([1,0,1,0,1,0,0,0,0], [0,1,-1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # M15 = (b+d+e) * (n - o)
    laderman_terms.append(([0,1,0,1,1,0,0,0,0], [0,0,0,0,1,-1,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # M16 = (d+e+g) * (o - p)
    laderman_terms.append(([0,0,0,1,1,0,1,0,0], [0,0,0,0,0,1,-1,0,0], [0,0,0,0,1,0,0,0,0]))
    # M17 = (d+e+h) * (q - r)
    laderman_terms.append(([0,0,0,1,1,0,0,1,0], [0,0,0,0,0,0,0,1,-1], [0,0,0,0,0,1,0,0,0]))
    # M18 = (f+h) * (m - p)
    laderman_terms.append(([0,0,0,0,0,1,0,1,0], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]))
    # M19 = (f+h) * (n - o)
    laderman_terms.append(([0,0,0,0,0,1,0,1,0], [0,0,0,0,1,-1,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # M20 = (f+h) * (p - q)
    laderman_terms.append(([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,0,1,-1,0], [0,0,1,0,0,0,0,0,0]))
    # M21 = (e+h) * (m - p)
    laderman_terms.append(([0,0,0,0,1,0,0,1,0], [0,0,0,1,0,0,-1,0,0], [0,0,0,0,1,0,0,0,0]))
    # M22 = (a+d+g) * (o - p)
    laderman_terms.append(([1,0,0,1,0,0,1,0,0], [0,0,0,0,0,1,-1,0,0], [1,0,0,0,0,0,0,0,0]))
    # M23 = (a+d+g) * (q - r)
    laderman_terms.append(([1,0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,1,-1], [1,0,0,0,0,0,0,0,0]))
    
    # Now build the full decomposition for 3x3 * 3x6
    # A is 3x3 (9 elements), B is 3x6 (18 elements), C is 3x6 (18 elements)
    # B = [B1 | B2] where B1, B2 are 3x3
    # B1 indices: 0..8 (row-major)
    # B2 indices: 9..17 (row-major)
    # C = [C1 | C2] where C1, C2 are 3x3
    # C1 = A * B1, C2 = A * B2
    
    nm = n * m  # 9
    mp = m * p  # 18
    np_ = p * n  # 18
    
    # We'll create rank-46 decomposition
    # First 23 terms for C1 = A * B1
    # Next 23 terms for C2 = A * B2
    
    U = [[0] * 46 for _ in range(nm)]
    V = [[0] * 46 for _ in range(mp)]
    W = [[0] * 46 for _ in range(np_)]
    
    # For C1 (columns 0..17 of C, but in our indexing C is p*n = 18 elements)
    # Wait, I need to be careful with indexing.
    # The output W has shape (p*n, rank) = (18, 46)
    # W[k*n + i, c] corresponds to C[i, k] in standard matrix multiplication
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    # In our tensor: T[x, y, z] where x = i*m+j, y = j'*p+k', z = k''*n+i''
    # T[x,y,z] = 1 if i=i'', j=j', k=k''
    
    # Let me rebuild this more carefully.
    # For the decomposition, each term c has:
    # U[:, c] is a vector of length nm=9 (for A's 9 elements)
    # V[:, c] is a vector of length mp=18 (for B's 18 elements)  
    # W[:, c] is a vector of length np_=18 (for C's 18 elements, indexed as k*n+i)
    
    # For C1 = A * B1 (first 3 columns of B, first 3 columns of C):
    # B1 occupies B indices 0..8 (j=0: B[0,0],B[0,1],B[0,2]; j=1: B[1,0],B[1,1],B[1,2]; j=2: B[2,0],B[2,1],B[2,2])
    # In row-major for 3x6: B[j,k] is at index j*6+k
    # So B1 elements are at indices: j*6+k for k=0,1,2
    # B2 elements are at indices: j*6+3+k for k=0,1,2
    
    # C1 = A * B1: C[i,k] for k=0,1,2
    # C2 = A * B2: C[i,k] for k=3,4,5
    # In W indexing: W[k*n+i, c] = C[i,k]
    # So C1 elements: W[k*3+i, c] for k=0,1,2
    # C2 elements: W[k*3+i, c] for k=3,4,5
    
    # Build first 23 terms for C1
    for t in range(23):
        alpha, beta, gamma = laderman_terms[t]
        col = t
        
        # U: alpha for A (9 elements)
        for a_idx in range(9):
            U[a_idx][col] = alpha[a_idx]
        
        # V: beta for B1 (only first 3 columns of B)
        # beta is for 3x3 B, so beta[b_idx] where b_idx = j*3+k for k=0,1,2
        # In the 3x6 B, this is at index j*6+k
        for j in range(3):
            for k in range(3):
                b_idx_3x3 = j * 3 + k
                b_idx_3x6 = j * 6 + k
                V[b_idx_3x6][col] = beta[b_idx_3x3]
        
        # W: gamma for C1 (first 3 columns of C)
        # gamma is for 3x3 C, so gamma[c_idx] where c_idx = i*3+k for k=0,1,2
        # In W indexing: W[k*3+i, col] = C[i,k]
        for i in range(3):
            for k in range(3):
                c_idx_3x3 = i * 3 + k
                w_idx = k * 3 + i  # k*n + i
                W[w_idx][col] = gamma[c_idx_3x3]
    
    # Build next 23 terms for C2
    for t in range(23):
        alpha, beta, gamma = laderman_terms[t]
        col = 23 + t
        
        # U: alpha for A (9 elements)
        for a_idx in range(9):
            U[a_idx][col] = alpha[a_idx]
        
        # V: beta for B2 (last 3 columns of B)
        # beta is for 3x3 B, so beta[b_idx] where b_idx = j*3+k for k=0,1,2
        # In the 3x6 B, this is at index j*6+3+k
        for j in range(3):
            for k in range(3):
                b_idx_3x3 = j * 3 + k
                b_idx_3x6 = j * 6 + 3 + k
                V[b_idx_3x6][col] = beta[b_idx_3x3]
        
        # W: gamma for C2 (last 3 columns of C)
        # gamma is for 3x3 C, so gamma[c_idx] where c_idx = i*3+k for k=0,1,2
        # In W indexing: W[k*3+i, col] = C[i,k] for k=3,4,5
        for i in range(3):
            for k in range(3):
                c_idx_3x3 = i * 3 + k
                w_idx = (3 + k) * 3 + i  # (k+3)*n + i
                W[w_idx][col] = gamma[c_idx_3x3]
    
    # Verify the decomposition
    def verify(U, V, W):
        nm, mp, np_ = len(U), len(V), len(W)
        rank = len(U[0]) if U else 0
        
        # Build target tensor
        T = np.zeros((nm, mp, np_))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    x = i * m + j
                    y = j * p + k
                    z = k * n + i
                    T[x, y, z] = 1.0
        
        # Compute decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        decomp = np.zeros((nm, mp, np_))
        for c in range(rank):
            decomp += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        
        return np.allclose(decomp, T, atol=1e-9)
    
    if verify(U, V, W):
        # Check if entries are integer or half-integer
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_check = np.round(U_arr * 2) / 2
        V_check = np.round(V_arr * 2) / 2
        W_check = np.round(W_arr * 2) / 2
        
        if np.allclose(U_check, U_arr, atol=1e-9) and np.allclose(V_check, V_arr, atol=1e-9) and np.allclose(W_check, W_arr, atol=1e-9):
            # Valid rank-46 decomposition
            return {"U": U, "V": V, "W": W}
    
    # If Laderman-based decomposition fails, fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
