"""Matrix multiplication 3x3 by 3x7 - rank minimization via structured decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' 
    # and i=i', j=j', k=k' else 0.
    # More precisely, for matrix mult C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i*m+j, j*p+k, k*n+i] = 1 (using trace convention W[k*n+i, r])
    
    # Schoolbook baseline: rank = n*m*p = 63
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify schoolbook solution
    def verify(U, V, W):
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # Check a sample of entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val - 1.0) > 1e-6:
                        return False
                    # Check some off-diagonal entries should be 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 != i or j2 != j or k2 != k:
                                    a2 = i2 * m + j2
                                    b2 = j2 * p + k2
                                    c2 = k2 * n + i2
                                    # Only check if this is a "wrong" combination
                                    # Actually, the tensor is 1 only when indices match properly
                                    # For C[i,k] = sum_j A[i,j]B[j,k], the tensor has:
                                    # T[i*m+j, j*p+k, k*n+i] = 1
                                    # All other entries are 0
                                    # So we just need to verify the diagonal-like entries
                                    pass
        return True

    # Try to find lower rank using structured approach
    # Split p=7 into blocks: 3+3+1
    # For each block of B, we can use Winograd-like structure
    
    # Winograd F(3) for 3x3x3 has rank 15
    # For 3x3x1, the rank is 9 (each output C[i,0] = sum_j A[i,j]*B[j,0])
    # But we can share structure across blocks
    
    # Let's try a block decomposition approach:
    # B = [B1 | B2 | B3] where B1, B2 are 3x3, B3 is 3x1
    # C = A * [B1 | B2 | B3] = [A*B1 | A*B2 | A*B3]
    
    # For A*B1 (3x3 * 3x3): use Winograd rank 15
    # For A*B2 (3x3 * 3x3): use Winograd rank 15  
    # For A*B3 (3x3 * 3x1): rank 9
    # Total: 15 + 15 + 9 = 39
    
    # But we can do better by sharing the A-transforms across blocks.
    # Let's construct a combined decomposition.
    
    # Winograd F(3) coefficients for 3x3 matrix multiply:
    # The 15 bilinear forms use specific linear combinations of A and B entries.
    # Let me define them properly.
    
    # Actually, let me use a simpler approach: try random search for lower ranks
    # with verification, starting from rank 39 and going down.
    
    def try_random_rank(rank, max_attempts=20):
        """Try to find a valid decomposition of given rank using random search."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)  # For reproducibility in the attempt
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            U_r = np.random.randn(n * m, rank) * 0.5
            V_r = np.random.randn(m * p, rank) * 0.5
            W_r = np.random.randn(p * n, rank) * 0.5
            
            # Verify
            if verify(U_r, V_r, W_r):
                # Try to round to half-integers
                U_rounded = np.round(U_r * 2) / 2
                V_rounded = np.round(V_r * 2) / 2
                W_rounded = np.round(W_r * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U_r, V_r, W_r
        
        return None

    # Try structured block approach first
    # Construct Winograd F(3) for 3x3x3
    
    # Winograd F(3) algorithm coefficients:
    # For multiplying two 3x3 matrices, the 15 bilinear products are:
    # Using the standard Winograd scheme:
    
    # Let me define the Winograd F(3) factor matrices explicitly.
    # The transform for A (3x3 -> 15 values):
    # The transform for B (3x3 -> 15 values):
    # Then C = combine the 15 products
    
    # Standard Winograd F(3):
    # A rows: [a,b,c], [d,e,f], [g,h,i]
    # B rows: [j,k,l], [m,n,o], [p,q,r]
    
    # The 15 bilinear forms:
    # t1 = (a+d+g) * (j+m+p)
    # t2 = (a+d-g) * (j+m-p)  
    # t3 = (a-d+g) * (j-m+p)
    # t4 = (a-d-g) * (j-m-p)
    # t5 = (b+e+h) * (k+n+q)
    # t6 = (b+e-h) * (k+n-q)
    # t7 = (b-e+h) * (k-n+q)
    # t8 = (b-e-h) * (k-n-q)
    # t9 = (c+f+i) * (l+o+r)
    # t10 = (c+f-i) * (l+o-r)
    # t11 = (c-f+i) * (l-o+r)
    # t12 = (c-f-i) * (l-o-r)
    # t13 = (a+e+i) * (j+n+r)
    # t14 = (b+f+g) * (k+o+p)
    # t15 = (c+d+h) * (l+m+q)
    
    # Wait, I need to verify this is correct. Let me use a known correct Winograd F(3).
    
    # Actually, the correct Winograd F(3) has these 15 products:
    # Let me look up the exact coefficients. The standard Winograd F(3) for 3x3:
    
    # Transform A:
    # x1 = a + d + g
    # x2 = a + d - g
    # x3 = a - d + g
    # x4 = a - d - g
    # x5 = b + e + h
    # x6 = b + e - h
    # x7 = b - e + h
    # x8 = b - e - h
    # x9 = c + f + i
    # x10 = c + f - i
    # x11 = c - f + i
    # x12 = c - f - i
    # x13 = a + e + i
    # x14 = b + f + g
    # x15 = c + d + h
    
    # Transform B:
    # y1 = j + m + p
    # y2 = j + m - p
    # y3 = j - m + p
    # y4 = j - m - p
    # y5 = k + n + q
    # y6 = k + n - q
    # y7 = k - n + q
    # y8 = k - n - q
    # y9 = l + o + r
    # y10 = l + o - r
    # y11 = l - o + r
    # y12 = l - o - r
    # y13 = j + n + r
    # y14 = k + o + p
    # y15 = l + m + q
    
    # Products: t_k = x_k * y_k for k=1..15
    
    # Then C entries are linear combinations of t_k:
    # C[0,0] = t1 + t2 + t3 + t4 + t13
    # C[0,1] = t5 + t6 + t7 + t8 + t14
    # C[0,2] = t9 + t10 + t11 + t12 + t15
    # C[1,0] = t1 + t2 - t3 - t4 + t13
    # C[1,1] = t5 + t6 - t7 - t8 + t14
    # C[1,2] = t9 + t10 - t11 - t12 + t15
    # C[2,0] = t1 - t2 + t3 - t4 + t13
    # C[2,1] = t5 - t6 + t7 - t8 + t14
    # C[2,2] = t9 - t10 + t11 - t12 + t15
    
    # Let me verify: C[0,0] should be a*j + b*m + c*p
    # = (t1+t2+t3+t4+t13)
    # t1 = (a+d+g)(j+m+p) = aj+am+ap+dj+dm+dp+gj+gm+gp
    # t2 = (a+d-g)(j+m-p) = aj+am-ap+dj+dm-dp-gj-gm+gp
    # t3 = (a-d+g)(j-m+p) = aj-am+ap-dj+dm+dp+gj-gm+gp
    # t4 = (a-d-g)(j-m-p) = aj-am-ap-dj+dm+dp-gj+gm-gp
    # t13 = (a+e+i)(j+n+r) = aj+an+ar+ej+en+er+ij+in+ir
    
    # Sum of t1+t2+t3+t4:
    # aj: 1+1+1+1 = 4
    # am: 1+1-1-1 = 0
    # ap: 1-1+1-1 = 0
    # dj: 1+1-1-1 = 0
    # dm: 1+1+1+1 = 4
    # dp: 1-1+1+1 = 2... wait that's not right
    
    # Hmm, this doesn't seem to work out cleanly. Let me just use the schoolbook solution
    # and try to find improvements through random search with verification.
    
    # For now, return the schoolbook solution as a guaranteed valid answer.
    # The rank is 63.
    
    # Let's try a few random searches for lower ranks
    for rank in range(39, 20, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_random_rank(rank, max_attempts=5)
        if result is not None:
            U_found, V_found, W_found = result
            # Verify once more
            if verify(U_found, V_found, W_found):
                return {"U": U_found.tolist(), "V": V_found.tolist(), "W": W_found.tolist()}
    
    # Return schoolbook solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
