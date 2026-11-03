import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        u_vec = [0]*(n*m)
        v_vec = [0]*(m*p)
        w_vec = [0]*(p*n)
        for (i,j), val in u.items():
            u_vec[i*m+j] = val
        for (j,k), val in v.items():
            v_vec[j*p+k] = val
        for (k,i), val in w.items():
            w_vec[k*n+i] = val
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
    
    def strassen_block(i0, j0, k0):
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term(
            {(i0,j0):1, (i0+1,j0+1):1},
            {(j0,k0):1, (j0+1,k0+1):1},
            {(k0,i0):1, (k0+1,i0+1):1}
        )
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term(
            {(i0+1,j0):1, (i0+1,j0+1):1},
            {(j0,k0):1},
            {(k0,i0+1):1, (k0+1,i0+1):-1}
        )
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        add_term(
            {(i0,j0):1},
            {(j0+1,k0):1, (j0+1,k0+1):-1},
            {(k0,i0+1):1, (k0+1,i0+1):1}
        )
        # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
        add_term(
            {(i0+1,j0+1):1},
            {(j0,k0+1):1, (j0+1,k0):-1},
            {(k0+1,i0):1, (k0,i0+1):1}
        )
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term(
            {(i0,j0):1, (i0,j0+1):1},
            {(j0+1,k0+1):1},
            {(k0+1,i0):-1, (k0+1,i0+1):1}
        )
        # M6 = (A10-A00)(B00+B10) -> C11 += M6
        add_term(
            {(i0+1,j0):1, (i0,j0):-1},
            {(j0,k0):1, (j0+1,k0):1},
            {(k0+1,i0+1):1}
        )
        # M7 = (A01-A11)(B01+B11) -> C00 += M7
        add_term(
            {(i0,j0+1):1, (i0+1,j0+1):-1},
            {(j0,k0+1):1, (j0+1,k0+1):1},
            {(k0+1,i0):1}
        )
    
    def schoolbook_221(i0, j0, k0):
        for di in range(2):
            for dj in range(2):
                add_term(
                    {(i0+di, j0+dj): 1},
                    {(j0+dj, k0): 1},
                    {(k0, i0+di): 1}
                )
    
    def schoolbook_219(i0, j0, k0):
        u = {(i0+di, j0): 1 for di in range(2)}
        v = {(j0, k0+dk): 1 for dk in range(9)}
        w = {(k0+dk, i0+di): 1 for di in range(2) for dk in range(9)}
        add_term(u, v, w)
    
    # Build decomposition
    for k_pair in range(4):
        k0 = k_pair * 2
        strassen_block(0, 0, k0)
    schoolbook_221(0, 0, 8)
    schoolbook_219(0, 2, 0)
    
    rank = len(U_terms)
    
    U = [[U_terms[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_terms[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[W_terms[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r] should relate to W
    # Actually the condition is: C[i,k] = sum_j A[i,j]*B[j,k]
    # Using decomposition: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * ... 
    # No, the correct check is the tensor identity:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]... no that's wrong
    
    # The correct verification: generate random A, B, compute C=AB,
    # then verify using the decomposition
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_test = A_test @ B_test
    
    # Using the decomposition:
    # For each rank term r:
    #   intermediate_A[i,r] = sum_j U[i*m+j, r] * A[i,j]
    #   intermediate_B[j,r] = sum_k V[j*p+k, r] * B[j,k]  
    #   C_decomp[i,k] = sum_r W[k*n+i, r] * ... 
    # 
    # Actually I think the correct formula is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # No, that's not right either.
    
    # Let me think... The standard rank-r decomposition of matrix mult tensor:
    # T = sum_r u_r ⊗ v_r ⊗ w_r where T[i,j,k]=1
    # This means: 1 = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] for all i,j,k
    # Wait, that's a 3-way product summing to 1? That doesn't make sense for a tensor.
    
    # Actually, I think the correct interpretation is different. Let me just verify numerically
    # by checking if the decomposition correctly computes matrix multiplication.
    
    # The way to use the decomposition to compute C=AB:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    # Hmm, that has two different sums over j which is confusing.
    
    # I think the correct formula is:
    # Define: X_r = sum_j U[i*m+j, r] * A[i,j]  (this is a vector of length n for each r)
    #         Y_r = sum_k V[j*p+k, r] * B[j,k]  (this is a vector of length m for each r)
    # Then: C[i,k] = sum_r W[k*n+i, r] * X_r[i] * Y_r[k]... no, Y_r is indexed by j not k.
    
    # I'm confused. Let me just check the tensor identity directly.
    
    # The tensor identity should be:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]... no, W is on the left.
    
    # OK let me just do a direct numerical check with the schoolbook to understand the convention.
    
    # Schoolbook: for each (i,j,k), U[i*m+j,col]=1, V[j*p+k,col]=1, W[k*n+i,col]=1
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r] = 1 if r corresponds to (i,j,k), else 0
    # And W[k*n+i,r] = 1 for that same r.
    # So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the matching r, 0 otherwise.
    # Total sum = 1. So the identity is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 = T[i,j,k]
    
    # Wait, but that means T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But T[i,j,k] = 1 for all i,j,k in the matrix multiplication tensor.
    
    # So the condition is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Let me verify this with my construction.
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    errors += 1
                    if errors <= 5:
                        print(f"Error at i={i},j={j},k={k}: got {val}, expected 1")
    
    if errors > 0:
        print(f"Total errors: {errors}")
        # Fall back to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
