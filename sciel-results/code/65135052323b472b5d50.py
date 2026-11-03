import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _build_strassen_inner():
    """
    Build U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
    Indices:
    U row: ii*2 + jj
    V row: jj*2 + kk
    W row: kk*2 + ii
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Term 0: M1
    # A: A00(1), A11(1) -> U[0,0]=1, U[3,0]=1
    # B: B00(1), B11(1) -> V[0,0]=1, V[3,0]=1
    # C: C00(1), C11(1) -> W[0,0]=1, W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2
    # A: A10(1), A11(1) -> U[2,1]=1, U[3,1]=1
    # B: B00(1) -> V[0,1]=1
    # C: C10(1) -> W[2,1]=1 (kk=1, ii=0 -> 1*2+0=2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 2: M3
    # A: A00(1) -> U[0,2]=1
    # B: B01(1), B11(-1) -> V[1,2]=1, V[3,2]=-1
    # C: C01(1) -> W[1,2]=1 (kk=0, ii=1 -> 0*2+1=1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 3: M4
    # A: A11(1) -> U[3,3]=1
    # B: B10(1), B00(-1) -> V[2,3]=1, V[0,3]=-1
    # C: C10(1) -> W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # Term 4: M5
    # A: A00(1), A01(1) -> U[0,4]=1, U[1,4]=1
    # B: B11(1) -> V[3,4]=1
    # C: C00(-1) -> W[0,4]=-1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    
    # Term 5: M6
    # A: A10(1), A00(-1) -> U[2,5]=1, U[0,5]=-1
    # B: B00(1), B01(1) -> V[0,5]=1, V[1,5]=1
    # C: C11(1) -> W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7
    # A: A01(1), A11(-1) -> U[1,6]=1, U[3,6]=-1
    # B: B10(1), B11(1) -> V[2,6]=1, V[3,6]=1
    # C: C00(1) -> W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_outer():
    """
    Build U, V, W for 2x2 * 2x3 = 2x3 using rank 11.
    Indices:
    U row: I*2 + J
    V row: J*3 + K
    W row: K*2 + I  (Note: Global W is k*n+i. Outer W index maps to block (I,K). 
                       In global space, block (I,K) contains elements. 
                       The outer W index should correspond to the block position.
                       Let's use W_out index = K*2 + I.
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part 1: Strassen for columns 0,1 of C (2x2 * 2x2)
    # This corresponds to B columns 0,1 and C columns 0,1.
    # U is same as Strassen U.
    # V uses B columns 0,1. V index J*3 + K. K in {0,1}.
    # W uses C columns 0,1. W index K*2 + I. K in {0,1}.
    
    U_str, V_str, W_str = _build_strassen_inner()
    
    # Copy U
    U[:, 0:7] = U_str
    
    # Map V_str (J*2 + K) to V_out (J*3 + K)
    for jj in range(2):
        for kk in range(2):
            v_in_idx = jj * 2 + kk
            v_out_idx = jj * 3 + kk
            V[v_out_idx, 0:7] = V_str[v_in_idx, 0:7]
            
    # Map W_str (K*2 + I) to W_out (K*2 + I) -- Indices are same!
    W[0:4, 0:7] = W_str[:, 0:7]
    
    # Part 2: Schoolbook for column 2 of C (2x2 * 2x1)
    # C[:, 2] = A * B[:, 2]
    # C0,2 = A0,0 B0,2 + A0,1 B1,2
    # C1,2 = A1,0 B0,2 + A1,1 B1,2
    
    # Term 7: A0,0 * B0,2 -> C0,2
    U[0, 7] = 1  # A0,0
    V[2, 7] = 1  # B0,2 (J=0, K=2 -> 0*3+2=2)
    W[2, 7] = 1  # C0,2 (K=2, I=0 -> 2*2+0=4? Wait.)
    
    # Let's re-verify W index mapping.
    # Global W row: k*n + i.
    # k = K*2 + kk, i = I*2 + ii.
    # Outer W index is K*2 + I.
    # Inner W index is kk*2 + ii.
    # Global W row = (K*2 + kk)*4 + (I*2 + ii) = K*8 + kk*4 + I*2 + ii.
    # In construct, we will iterate K, I, kk, ii.
    # W_global[ (K*2+kk)*4 + (I*2+ii), t*7+s ] = W_out[K*2+I, t] * W_in[kk*2+ii, s].
    
    # So for Term 7:
    # C0,2 means I=0, K=2.
    # W_out index = K*2 + I = 2*2 + 0 = 4.
    W[4, 7] = 1
    
    # Term 8: A0,1 * B1,2 -> C0,2
    U[1, 8] = 1  # A0,1
    V[5, 8] = 1  # B1,2 (J=1, K=2 -> 1*3+2=5)
    W[4, 8] = 1  # C0,2 (K=2, I=0 -> 4)
    
    # Term 9: A1,0 * B0,2 -> C1,2
    U[2, 9] = 1  # A1,0
    V[2, 9] = 1  # B0,2
    W[5, 9] = 1  # C1,2 (K=2, I=1 -> 2*2+1=5)
    
    # Term 10: A1,1 * B1,2 -> C1,2
    U[3, 10] = 1 # A1,1
    V[5, 10] = 1 # B1,2
    W[5, 10] = 1 # C1,2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    U_out, V_out, W_out = _build_outer()
    U_in, V_in, W_in = _build_strassen_inner()
    
    rank_out = 11
    rank_in = 7
    total_rank = rank_out * rank_in
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Outer indices:
    # I in 0..1, J in 0..1
    # J in 0..1, K in 0..2
    # K in 0..2, I in 0..1
    
    # Inner indices:
    # ii in 0..1, jj in 0..1
    # jj in 0..1, kk in 0..1
    # kk in 0..1, ii in 0..1
    
    for t in range(rank_out):
        for s in range(rank_in):
            col = t * rank_in + s
            
            for I in range(2):
                for J in range(2):
                    # U part
                    u_out_val = U_out[I * 2 + J, t]
                    if u_out_val == 0: continue
                    for ii in range(2):
                        for jj in range(2):
                            u_in_val = U_in[ii * 2 + jj, s]
                            if u_in_val == 0: continue
                            
                            # Global U row: i*m + j = (I*2+ii)*4 + (J*2+jj)
                            i = I * 2 + ii
                            j = J * 2 + jj
                            u_global_row = i * m + j
                            U[u_global_row, col] += u_out_val * u_in_val
                            
            for J in range(2):
                for K in range(3):
                    # V part
                    v_out_val = V_out[J * 3 + K, t]
                    if v_out_val == 0: continue
                    for jj in range(2):
                        for kk in range(2):
                            v_in_val = V_in[jj * 2 + kk, s]
                            if v_in_val == 0: continue
                            
                            # Global V row: j*p + k = (J*2+jj)*6 + (K*2+kk)
                            j = J * 2 + jj
                            k = K * 2 + kk
                            v_global_row = j * p + k
                            V[v_global_row, col] += v_out_val * v_in_val
                            
            for K in range(3):
                for I in range(2):
                    # W part
                    w_out_val = W_out[K * 2 + I, t]
                    if w_out_val == 0: continue
                    for kk in range(2):
                        for ii in range(2):
                            w_in_val = W_in[kk * 2 + ii, s]
                            if w_in_val == 0: continue
                            
                            # Global W row: k*n + i = (K*2+kk)*4 + (I*2+ii)
                            k = K * 2 + kk
                            i = I * 2 + ii
                            w_global_row = k * n + i
                            W[w_global_row, col] += w_out_val * w_in_val
                            
    # Convert to lists of floats (or ints if possible)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Optional: Verify
    # err = _verify(U_list, V_list, W_list, n, m, p, total_rank)
    # if err > 1e-6:
    #     # Fallback
    #     pass
        
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err
