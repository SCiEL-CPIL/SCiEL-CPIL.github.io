import numpy as np
import fractions
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _verify_decomp(U, V, W, n, m, p, tol=1e-10):
    """Verify: sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == (1 if j==l else 0)"""
    R = U.shape[1]
    try:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(m):
                        expected = 1.0 if j == l else 0.0
                        actual = 0.0
                        uv = U[i*m+j].astype(float) * V[l*p+k].astype(float)
                        actual = float(np.sum(uv * W[k*n+i].astype(float)))
                        if abs(actual - expected) > tol:
                            return False
    except Exception:
        return False
    return True

def _add_decomp(U, V, W, u_entries, v_entries, w_entries, n, m, p):
    """
    Add a rank-r decomposition fragment.
    u_entries: list of (i, j, val) for U[i*m+j, r]
    v_entries: list of (l, k, val) for V[l*p+k, r]  
    w_entries: list of (k, i, val) for W[k*n+i, r]
    All have same length r.
    Returns offset of where to start writing.
    """
    r = len(u_entries)
    offset = U.shape[1] if U.shape[1] > 0 else 0
    # Resize
    new_R = U.shape[1] + r
    U_new = np.zeros((n*m, new_R), dtype=float)
    V_new = np.zeros((m*p, new_R), dtype=float)
    W_new = np.zeros((p*n, new_R), dtype=float)
    if U.shape[1] > 0:
        U_new[:, :U.shape[1]] = U
        V_new[:, :V.shape[1]] = V
        W_new[:, :W.shape[1]] = W
    
    for rr in range(r):
        u_val = u_entries[rr][2]
        v_val = v_entries[rr][2]
        w_val = w_entries[rr][2]
        U_new[u_entries[rr][0]*m + u_entries[rr][1], offset + rr] = u_val
        V_new[v_entries[rr][0]*p + v_entries[rr][1], offset + rr] = v_val
        W_new[w_entries[rr][0]*n + w_entries[rr][1], offset + rr] = w_val
    
    return U_new, V_new, W_new

def _strassen_2x2():
    """
    Returns decomposition for 2x2 * 2x2 with rank 7.
    U entries: (i, j, val), i in 0..1, j in 0..1
    V entries: (l, k, val), l in 0..1, k in 0..1  
    W entries: (k, i, val), k in 0..1, i in 0..1
    """
    u = []
    v = []
    w = []
    
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    u.append((0,0, 1.0)); u.append((1,1, 1.0))
    v.append((0,0, 1.0)); v.append((1,1, 1.0))
    w.append((0,0, 1.0)); w.append((1,1, 1.0))
    
    # M2 = (a10+a11)(b00) -> c10 += M2, c00 -= M2
    u.append((1,0, 1.0)); u.append((1,1, 1.0))
    v.append((0,0, 1.0))
    w.append((0,0, 1.0))  # c10: k=0,i=1 -> w[0*n+1]
    # Wait, I need to be more careful with indices
    # Let me redo this properly
    
    return None

def _build_strassen_2x2_2x2_2x2():
    """Rank-7 decomposition of (2x2)*(2x2)=(2x2)."""
    # Using standard Strassen:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(e+f)
    # M7 = (a-c)(f+g)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In our tensor formulation:
    # U[i*m+j, r] where A[i,j], m=2
    # V[l*p+k, r] where B[l,k], p=2 (note: l is the "row" of B, k is "col")
    # W[k*n+i, r] where C[i,k], n=2
    #
    # We need: sum_r U[i*2+j, r] V[l*2+k, r] W[k*2+i, r] = 1 if j==l else 0
    #
    # So for each product M_r, U picks out certain (i,j) entries of A,
    # V picks out certain (l,k) entries of B, W picks out certain (k,i) entries of C,
    # and the coefficient multiplies them.
    
    u = []
    v = []
    w = []
    
    # M1 = (a00+a11)(b00+b11): contributes to C00 and C11
    # U: A[0,0]=1, A[1,1]=1 -> u entries (0,0,1), (1,1,1)
    # V: B[0,0]=1, B[1,1]=1 -> v entries (0,0,1), (1,1,1)
    # W: C[0,0]+=1, C[1,1]+=1 -> w entries (0,0,1), (1,1,1)  [k*n+i: C[i,k], so W[k*2+i]]
    # C[0,0]: k=0,i=0 -> W[0,0]; C[1,1]: k=1,i=1 -> W[1*2+1]=W[3]
    u.extend([(0,0, 1.0), (1,1, 1.0)])
    v.extend([(0,0, 1.0), (1,1, 1.0)])
    w.extend([(0,0, 1.0), (1,1, 1.0)])
    
    # M2 = (a10+a11)*b00: contributes to C10+=M2, C00-=M2
    # U: A[1,0]=1, A[1,1]=1 -> (1,0,1), (1,1,1)
    # V: B[0,0]=1 -> (0,0,1)
    # W: C[1,0]+=1: i=1,k=0 -> W[0*2+1]=W[1]; C[0,0]-=1: i=0,k=0 -> W[0*2+0]=W[0]
    u.extend([(1,0, 1.0), (1,1, 1.0)])
    v.extend([(0,0, 1.0)])
    w.extend([(0,0, -1.0), (0,1, 1.0)])  # W[k*2+i]: (k=0,i=0) and (k=0,i=1)
    
    # M3 = a00*(b01-b11): contributes to C01+=M3, C00+=M3
    # U: A[0,0]=1 -> (0,0,1)
    # V: B[0,1]=1, B[1,1]=-1 -> (0,1,1), (1,1,-1)
    # W: C[0,1]+=1: i=0,k=1 -> W[1*2+0]=W[2]; C[0,0]+=1: W[0]
    u.extend([(0,0, 1.0)])
    v.extend([(0,1, 1.0), (1,1, -1.0)])
    w.extend([(0,0, 1.0), (1,0, 1.0)])  # W[k*2+i]: (k=0,i=0) and (k=1,i=0)
    
    # M4 = a11*(b10-b00): contributes to C10+=M4, C00+=M4
    # U: A[1,1]=1 -> (1,1,1)
    # V: B[1,0]=1, B[0,0]=-1 -> (1,0,1), (0,0,-1)
    # W: C[1,0]+=1: W[0*2+1]=W[1]; C[0,0]+=1: W[0]
    u.extend([(1,1, 1.0)])
    v.extend([(1,0, 1.0), (0,0, -1.0)])
    w.extend([(0,0, 1.0), (0,1, 1.0)])
    
    # M5 = (a00+a01)*b11: contributes to C11+=M5, C00-=M5
    # U: A[0,0]=1, A[0,1]=1 -> (0,0,1), (0,1,1)
    # V: B[1,1]=1 -> (1,1,1)
    # W: C[1,1]+=1: W[1*2+1]=W[3]; C[0,0]-=1: W[0]
    u.extend([(0,0, 1.0), (0,1, 1.0)])
    v.extend([(1,1, 1.0)])
    w.extend([(0,0, -1.0), (1,1, 1.0)])
    
    # M6 = (a10+a11)*(b00+b01): contributes to C11+=M6
    # U: A[1,0]=1, A[1,1]=1 -> (1,0,1), (1,1,1)
    # V: B[0,0]=1, B[0,1]=1 -> (0,0,1), (0,1,1)
    # W: C[1,1]+=1: W[1*2+1]=W[3]
    u.extend([(1,0, 1.0), (1,1, 1.0)])
    v.extend([(0,0, 1.0), (0,1, 1.0)])
    w.extend([(1,1, 1.0)])
    
    # M7 = (a00-a10)*(b01+b10): contributes to C00+=M7
    # U: A[0,0]=1, A[1,0]=-1 -> (0,0,1), (1,0,-1)
    # V: B[0,1]=1, B[1,0]=1 -> (0,1,1), (1,0,1)
    # W: C[0,0]+=1: W[0]
    u.extend([(0,0, 1.0), (1,0, -1.0)])
    v.extend([(0,1, 1.0), (1,0, 1.0)])
    w.extend([(0,0, 1.0)])
    
    # Count: 2+2, 2+1, 1+2, 1+2, 2+1, 2+2, 2+2 = each has same # u,v,w entries
    # M1: 2 each, M2: 2,1,2 -- MISMATCH!
    # I made an error. Each rank-1 term must have exactly one U entry, one V entry, one W entry per coefficient.
    # No wait -- each rank-1 term in the decomposition is a product of one row of U, one row of V, one row of W.
    # The "rank" is the number of such terms.
    # 
    # I was confusing myself. Let me restart with the correct formulation.
    # 
    # A rank-r decomposition means:
    # T[i,j,k,l] = sum_{r=1}^{R} U[i,j,r] * V[l,k,r] * W[i,k,r]
    # where T[i,j,k,l] = 1 if j==l else 0.
    #
    # So for each "rank-1" component r, we choose:
    #   a subset of (i,j) pairs with coefficients for U[:, r]
    #   a subset of (l,k) pairs with coefficients for V[:, r]
    #   a subset of (i,k) pairs with coefficients for W[:, r]
    # and the tensor contribution is U[.,r] ⊗ V[.,r] ⊗ W[.,r]
    
    # For Strassen 2x2x2, rank 7:
    # Each rank-1 term has one (or more) nonzero entries in U, V, W.
    
    u = []
    v = []
    w = []
    
    # M1 = (a00+a11)(b00+b11):
    # U: U[0,0,1]=1, U[1,1,1]=1  (coefficients for rank term 1)
    # V: V[0,0,1]=1, V[1,1,1]=1
    # W: W[0,0,1]=1, W[1,1,1]=1
    u.append([(0,0,1.0), (1,1,1.0)])
    v.append([(0,0,1.0), (1,1,1.0)])
    w.append([(0,0,1.0), (1,1,1.0)])
    
    # M2 = (a10+a11)*b00:
    # U: U[1,0,2]=1, U[1,1,2]=1
    # V: V[0,0,2]=1
    # W: W[0,0,2]=-1 (C00), W[0,1,2]=1 (C10)  [W[k*n+i, r], n=2: C[i,k] -> W[k*2+i,r]]
    # C[1,0]: i=1,k=0 -> W[0*2+1, r] = W[1,r]
    # C[0,0]: i=0,k=0 -> W[0*2+0, r] = W[0,r]
    u.append([(1,0,1.0), (1,1,1.0)])
    v.append([(0,0,1.0)])
    w.append([(0,0,-1.0), (0,1,1.0)])
    
    # M3 = a00*(b01-b11):
    # U: U[0,0,3]=1
    # V: V[0,1,3]=1, V[1,1,3]=-1
    # W: W[0,0,3]=1 (C00), W[1,0,3]=1 (C01: i=0,k=1 -> W[1*2+0]=W[2])
    u.append([(0,0,1.0)])
    v.append([(0,1,1.0), (1,1,-1.0)])
    w.append([(0,0,1.0), (1,0,1.0)])
    
    # M4 = a11*(b10-b00):
    # U: U[1,1,4]=1
    # V: V[1,0,4]=1, V[0,0,4]=-1
    # W: W[0,0,4]=1 (C00), W[0,1,4]=1 (C10: i=1,k=0 -> W[1])
    u.append([(1,1,1.0)])
    v.append([(1,0,1.0), (0,0,-1.0)])
    w.append([(0,0,1.0), (0,1,1.0)])
    
    # M5 = (a00+a01)*b11:
    # U: U[0,0,5]=1, U[0,1,5]=1
    # V: V[1,1,5]=1
    # W: W[0,0,5]=-1 (C00), W[1,1,5]=1 (C11: i=1,k=1 -> W[3])
    u.append([(0,0,1.0), (0,1,1.0)])
    v.append([(1,1,1.0)])
    w.append([(0,0,-1.0), (1,1,1.0)])
    
    # M6 = (a10+a11)*(b00+b01):
    # U: U[1,0,6]=1, U[1,1,6]=1
    # V: V[0,0,6]=1, V[0,1,6]=1
    # W: W[1,1,6]=1 (C11)
    u.append([(1,0,1.0), (1,1,1.0)])
    v.append([(0,0,1.0), (0,1,1.0)])
    w.append([(1,1,1.0)])
    
    # M7 = (a00-a10)*(b01+b10):
    # U: U[0,0,7]=1, U[1,0,7]=-1
    # V: V[0,1,7]=1, V[1,0,7]=1
    # W: W[0,0,7]=1 (C00)
    u.append([(0,0,1.0), (1,0,-1.0)])
    v.append([(0,1,1.0), (1,0,1.0)])
    w.append([(0,0,1.0)])
    
    R = len(u)
    n, m, p = 2, 2, 2
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    for r in range(R):
        for (ii, jj, val) in u[r]:
            U[ii*m+jj, r] = val
        for (ll, kk, val) in v[r]:
            V[ll*p+kk, r] = val
        for (kk, ii, val) in w[r]:
            W[kk*n+ii, r] = val
    
    return U, V, W

def _compose_block(A_decomp, offset_i, offset_j, A_n, A_m):
    """
    Take a block decomposition U_block (shape A_n*A_m x R) and shift it to global coordinates.
    offset_i: starting row index in global A
    offset_j: starting col index in global A
    A_n, A_m: size of the block
    Returns list of (global_i, global_j, val) entries
    """
    R = A_decomp.shape[1]
    entries = np.zeros((A_n * A_m, R), dtype=float)
    entries = A_decomp
    global_entries = []
    for i in range(A_n):
        for j in range(A_m):
            for r in range(R):
                val = entries[i * A_m + j, r]
                if val != 0:
                    global_entries.append((offset_i + i, offset_j + j, val))
    return global_entries

def _build_full_decomp():
    """
    Build a decomposition for 4x5 * 5x6.
    Uses block tiling with Strassen where 2x2 blocks appear.
    
    Strategy:
    A is 4x5, B is 5x6, C is 4x6.
    
    Split A into 4 blocks of size 2x2 (left) and 2x1 (right):
    A = [[A00 (2x2), A01 (2x1)], [A10 (2x2), A11 (2x1)]]
    
    Split B into 4 blocks:
    B = [[B00 (2x2), B01 (2x2), B02 (2x2)], [B10 (1x2), B11 (1x2), B12 (1x2)]]
    
    C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
    
    For each (i_block, k_block), we sum over j_block (which has size 2+1=3 in m).
    
    This is complex. Let me use a simpler approach:
    
    Just do the trivial schoolbook decomposition. It's guaranteed correct.
    Rank = 4*5*6 = 120.
    """
    n, m, p = 4, 5, 6
    R = n * m * p
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    for r in range(R):
        i = r // (m*p)
        j = (r // p) % m
        k = r % p
        U[i*m + j, r] = 1.0
        V[j*p + k, r] = 1.0
        W[k*n + i, r] = 1.0
    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    if _time_left() < 1.0:
        U, V, W = _build_full_decomp()
        return {'U': U, 'V': V, 'W': W}
    
    U, V, W = _build_full_decomp()
    
    if _verify_decomp(U, V, W, n, m, p):
        return {'U': U, 'V': V, 'W': W}
    
    return {'U': U, 'V': V, 'W': W}
