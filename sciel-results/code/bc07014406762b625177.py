DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-minimal decomposition using structured factors.
    # For n=2, m=3, p=10: schoolbook rank is 60.
    # We try rank r and verify.
    # Strategy: use the fact that n is small.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We need factors U (6 x r), V (30 x r), W (20 x r) such that
    # for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta-ish
    # Actually the tensor equation is:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    
    # For n=2,m=3,p=10, try to use a Kronecker-product-like structure.
    # Since n=2 is prime and small, try rank = m*p + m*p - something?
    # Let's just try the schoolbook first as a valid baseline, then attempt reduction.
    
    # Actually, let me try a direct approach: rank = 30
    # Use the identity: for each pair (j,k), we need to cover both i=0 and i=1.
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # One idea: for each j in {0,1,2} and k in {0,...,9}, introduce a factor that
    # handles B[j,k] times (A[0,j], A[1,j]). This is exactly schoolbook rank 60.
    
    # Can we share? If we could factor A[0,j] and A[1,j] into shared terms...
    # For a fixed j, we need to produce two outputs (for i=0 and i=1) from the same B[j,k].
    # That's 2 terms per (j,k) pair, so 60 total. Unless there's algebraic cancellation.
    
    # Let me try rank 40 or 30 with random initialization and check.
    # For now, return schoolbook (rank 60) which is known valid.
    # But let's try to do better: use rank = 3*p*n = 60... no.
    
    # Try a different structure: group k values.
    # For each i (0,1) and j (0,1,2), we have 10 outputs k.
    # C[i,k] depends on all j. 
    
    # Let's just return the schoolbook decomposition which is guaranteed valid.
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
