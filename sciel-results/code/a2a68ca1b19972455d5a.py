import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    
    def verify(U, V, W, R):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                c = k2 * n + i2
                                # b should be j2*p+k2, but we fixed b=j*p+k
                                # So we need j2==j and k2==k for the tensor to be potentially nonzero
                                if j2 != j or k2 != k:
                                    continue
                                # Now check if i2 != i
                                if i2 != i:
                                    val = 0.0
                                    for r in range(R):
                                        val += U[a][r] * V[b][r] * W[c][r]
                                    if abs(val) > 1e-9:
                                        return False
        return True
    
    def strassen_2x2x2(offset_a, offset_b, offset_c):
        """
        Return list of (U_vec, V_vec, W_vec) for 2x2x2 Strassen.
        offset_a: column offset in A (A is n x m, we're using cols offset_a, offset_a+1)
        offset_b: row offset in B (B is m x p, we're using rows offset_b, offset_b+1)  
        offset_c: column offset in C (C is n x p, we're using cols offset_c, offset_c+1)
        """
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 at (0, offset_a), A22 at (1, offset_a+1)
        u1 = [0]*(n*m); u1[0*m + offset_a] = 1; u1[1*m + (offset_a+1)] = 1
        v1 = [0]*(m*p); v1[offset_b*p + offset_c] = 1; v1[(offset_b+1)*p + (offset_c+1)] = 1
        w1 = [0]*(p*n); w1[offset_c*n + 0] = 1; w1[(offset_c+1)*n + 1] = 1
        terms.append((u1, v1, w1))
        
        # M2 = (A21+A22)B11
        u2 = [0]*(n*m); u2[1*m + offset_a] = 1; u2[1*m + (offset_a+1)] = 1
        v2 = [0]*(m*p); v2[offset_b*p + offset_c] = 1
        w2 = [0]*(p*n); w2[offset_c*n + 1] = 1; w2[(offset_c+1)*n + 1] = -1
        terms.append((u2, v2, w2))
        
        # M3 = A11(B12-B22)
        u3 = [0]*(n*m); u3[0*m + offset_a] = 1
        v3 = [0]*(m*p); v3[offset_b*p + (offset_c+1)] = 1; v3[(offset_b+1)*p + (offset_c+1)] = -1
        w3 = [0]*(p*n); w3[(offset_c+1)*n + 0] = 1; w3[(offset_c+1)*n + 1] = 1
        terms.append((u3, v3, w3))
        
        # M4 = A22(B21-B11)
        u4 = [0]*(n*m); u4[1*m + (offset_a+1)] = 1
        v4 = [0]*(m*p); v4[(offset_b+1)*p + offset_c] = 1; v4[offset_b*p + offset_c] = -1
        w4 = [0]*(p*n); w4[offset_c*n + 0] = 1; w4[offset_c*n + 1] = 1
        terms.append((u4, v4, w4))
        
        # M5 = (A11+A12)B22
        u5 = [0]*(n*m); u5[0*m + offset_a] = 1; u5[0*m + (offset_a+1)] = 1
        v5 = [0]*(m*p); v5[(offset_b+1)*p + (offset_c+1)] = 1
        w5 = [0]*(p*n); w5[offset_c*n + 0] = -1; w5[(offset_c+1)*n + 0] = 1
        terms.append((u5, v5, w5))
        
        # M6 = (A21-A11)(B11+B12)
        u6 = [0]*(n*m); u6[1*m + offset_a] = 1; u6[0*m + offset_a] = -1
        v6 = [0]*(m*p); v6[offset_b*p + offset_c] = 1; v6[offset_b*p + (offset_c+1)] = 1
        w6 = [0]*(p*n); w6[(offset_c+1)*n + 1] = 1
        terms.append((u6, v6, w6))
        
        # M7 = (A12-A22)(B21+B22)
        u7 = [0]*(n*m); u7[0*m + (offset_a+1)] = 1; u7[1*m + (offset_a+1)] = -1
        v7 = [0]*(m*p); v7[(offset_b+1)*p + offset_c] = 1; v7[(offset_b+1)*p + (offset_c+1)] = 1
        w7 = [0]*(p*n); w7[offset_c*n + 0] = 1
        terms.append((u7, v7, w7))
        
        return terms
    
    # Try block decomposition: split m=5 into parts
    # For each part of size s, we compute A[:, off:off+s] * B[off:off+s, :]
    # If s=2 and we can split p into 2's, use Strassen
    # If s=3, use schoolbook or find better
    
    def try_split(split_sizes):
        """Try a specific split of m into parts."""
        if sum(split_sizes) != m:
            return None
        
        all_U = []
        all_V = []
        all_W = []
        
        a_offset = 0  # column offset in A
        b_offset = 0  # row offset in B
        
        for s in split_sizes:
            if s == 2:
                # Use Strassen for 2x2x2 blocks
                # Split p into chunks of 2
                for c_off in range(0, p, 2):
                    if c_off + 2 > p:
                        # Handle remainder with schoolbook
                        for i in range(n):
                            for j in range(2):
                                for k in range(p - c_off):
                                    u = [0]*(n*m); u[i*m + (a_offset+j)] = 1
                                    v = [0]*(m*p); v[(b_offset+j)*p + (c_off+k)] = 1
                                    w = [0]*(p*n); w[(c_off+k)*n + i] = 1
                                    all_U.append(u); all_V.append(v); all_W.append(w)
                    else:
                        terms = strassen_2x2x2(a_offset, b_offset, c_off)
                        for u, v, w in terms:
                            all_U.append(u); all_V.append(v); all_W.append(w)
            elif s == 1:
                for i in range(n):
                    for k in range(p):
                        u = [0]*(n*m); u[i*m + a_offset] = 1
                        v = [0]*(m*p); v[b_offset*p + k] = 1
                        w = [0]*(p*n); w[k*n + i] = 1
                        all_U.append(u); all_V.append(v); all_W.append(w)
            elif s == 3:
                # Schoolbook for 2x3xp
                for i in range(n):
                    for j in range(3):
                        for k in range(p):
                            u = [0]*(n*m); u[i*m + (a_offset+j)] = 1
                            v = [0]*(m*p); v[(b_offset+j)*p + k] = 1
                            w = [0]*(p*n); w[k*n + i] = 1
                            all_U.append(u); all_V.append(v); all_W.append(w)
            else:
                # General schoolbook
                for i in range(n):
                    for j in range(s):
                        for k in range(p):
                            u = [0]*(n*m); u[i*m + (a_offset+j)] = 1
                            v = [0]*(m*p); v[(b_offset+j)*p + k] = 1
                            w = [0]*(p*n); w[k*n + i] = 1
                            all_U.append(u); all_V.append(v); all_W.append(w)
            
            a_offset += s
            b_offset += s
        
        R = len(all_U)
        if verify(all_U, all_V, all_W, R):
            return (R, all_U, all_V, all_W)
        return None
    
    # Try various splits
    splits_to_try = [
        [2, 3],
        [3, 2],
        [2, 2, 1],
        [1, 2, 2],
        [2, 1, 2],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2],
        [2, 2, 1],
    ]
    
    for splits in splits_to_try:
        if time.time() > deadline:
            break
        result = try_split(splits)
        if result is not None:
            R, U, V, W = result
            if R < best_rank:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
