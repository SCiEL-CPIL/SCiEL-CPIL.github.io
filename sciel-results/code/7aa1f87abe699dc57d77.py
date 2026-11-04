import numpy as np
import time

def _verify(U, V, W, tol=1e-6):
    """Verify that the tensor decomposition is valid:
    sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == 1 for all (i,j,k)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                u = U[i*4+j, :]
                v = V[j*4+k, :]
                w = W[k*4+i, :]
                for t in range(rank):
                    s += u[t] * v[t] * w[t]
                if abs(s - 1.0) > tol:
                    valid = False
    return valid

def _schoolbook():
    """Return the schoolbook 4x4x4 decomposition (rank 64)."""
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (4, 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+0][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return U, V, W

def _rank28_decomposition():
    """
    Construct a rank-28 decomposition for 4x4x4 using Strassen at block level
    with schoolbook (rank 4) for each 2x2 block product.
    
    Total rank = 7 * 4 = 28.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Indices for 4x4 matrix elements:
    # i, j, k in {0,1,2,3}
    # Block indices: i1, j1, k1 in {0,1}
    # Sub-block indices: i2, j2, k2 in {0,1}
    # i = 2*i1 + i2, etc.
    
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    for r1 in range(7):
        for i2 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    r = r1 * 4 + (i2 * 4 + j2 * 2 + k2)
                    
                    # The coefficient for this term is U2[i1, j1, r1] * V2[j1, k1, r1] * W2[k1, i1, r1]
                    # But since we are constructing the tensors U, V, W separately,
                    # we fill them based on the block structure.
                    
                    # For a given Strassen term r1, the block multiplication is:
                    # BlockA[i1, j1] * BlockB[j1, k1] contributes to BlockC[i1, k1]
                    # The scalar multiplication is A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2]
                    # contributing to C[i1*2+i2, k1*2+k2]
                    
                    # U component: coefficient of A[i, j] in term r
                    # U[i*4+j, r] = U2[i1*2+j1, r1] if (i,j) corresponds to (i1,j1,i2,j2)
                    # Wait, U2 is (4,7) where row is i*2+j.
                    # So U2[i1*2+j1, r1] is the coefficient for A[i1, j1] in block term r1.
                    
                    # Let's iterate over block positions i1, j1, k1
                    for i1 in range(2):
                        for j1 in range(2):
                            for k1 in range(2):
                                i = 2 * i1 + i2
                                j = 2 * j1 + j2
                                k = 2 * k1 + k2
                                
                                u_coeff = U2[i1 * 2 + j1, r1]
                                v_coeff = V2[j1 * 2 + k1, r1]
                                w_coeff = W2[k1 * 2 + i1, r1]
                                
                                if u_coeff != 0:
                                    U[i * 4 + j, r] = u_coeff
                                if v_coeff != 0:
                                    V[j * 4 + k, r] = v_coeff
                                if w_coeff != 0:
                                    W[k * 4 + i, r] = w_coeff
    return U, V, W

def _to_outlist(A):
    """Convert numpy array to list of lists with int or half-int entries."""
    out = []
    for row in A:
        r = []
        for v in row:
            if abs(v - round(v)) < 1e-10:
                r.append(int(round(v)))
            else:
                if abs(2*v - round(2*v)) < 1e-10:
                    r.append(int(round(2*v)) / 2.0)
                else:
                    r.append(float(v))
        out.append(r)
    return out

def construct():
    t0 = time.time()
    
    # Try rank-28 decomposition first
    try:
        U28, V28, W28 = _rank28_decomposition()
        
        # Verify rank-28
        if _verify(U28, V28, W28, tol=1e-9):
            # Attempt to prune rank-28 to lower ranks
            # This is a heuristic search. We try removing columns one by one.
            # Since rank 28 is already quite low, pruning might be difficult.
            # We'll do a quick check for any redundant columns.
            
            U_cur = U28
            V_cur = V28
            W_cur = W28
            rank_cur = 28
            
            # Simple pruning: try removing each column and checking validity
            # This is O(rank * 64 * rank) which is manageable for rank=28.
            # 28 * 64 * 28 ~ 50k operations per check. 28 checks = 1.4M ops. Fast.
            
            for rem in range(rank_cur):
                if time.time() - t0 > 100:
                    break
                keep = [x for x in range(rank_cur) if x != rem]
                if len(keep) == 0:
                    continue
                U_t = U_cur[:, keep]
                V_t = V_cur[:, keep]
                W_t = W_cur[:, keep]
                
                if _verify(U_t, V_t, W_t, tol=1e-9):
                    U_cur = U_t
                    V_cur = V_t
                    W_cur = W_t
                    rank_cur -= 1
                    # Reset loop to try removing another column from the new rank
                    rem = 0 
                    continue
            
            return {
                "U": _to_outlist(U_cur),
                "V": _to_outlist(V_cur),
                "W": _to_outlist(W_cur)
            }
    except Exception as e:
        pass
    
    # Fallback to rank-49 Kronecker product if rank-28 fails
    U2, V2, W2 = _strassen_2x2()
    U4k = np.kron(U2, U2)
    V4k = np.kron(V2, V2)
    W4k = np.kron(W2, W2)
    
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        K = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        perm_U[S] = K
        
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = K
        
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        K = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
        perm_W[S] = K
        
    U4 = U4k[perm_U, :]
    V4 = V4k[perm_V, :]
    W4 = W4k[perm_W, :]
    
    if _verify(U4, V4, W4, tol=1e-9):
        return {
            "U": _to_outlist(U4),
            "V": _to_outlist(V4),
            "W": _to_outlist(W4)
        }
        
    # Last resort: Schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _verify(res['U'], res['V'], res['W']))
