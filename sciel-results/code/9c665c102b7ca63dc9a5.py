import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _generate_tensor(n, m, p):
    """Generate the matrix multiplication tensor T[a,b,c] for n x m * m x p -> n x p.
    Indices: a = i*m + j, b = j'*p + k, c = k'*n + i'
    T[a,b,c] = 1 iff j == j' and i == i' and k == k'
    """
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if j == j2 and i == i2 and k == k2:
                                a = i * m + j
                                b = j2 * p + k
                                c = k2 * n + i2
                                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify if U, V, W decompose the tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Efficient verification using tensor contraction
    # T_reconstructed[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We can check specific entries or the whole tensor.
    # For small dimensions, checking the whole tensor is fine.
    
    # Generate expected tensor
    T_expected = _generate_tensor(n, m, p)
    
    # Reconstruct
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if not np.allclose(T_recon, T_expected, atol=tol):
        return False
    return True

def _als_decompose(T, n, m, p, rank, max_iter=1000, tol=1e-6, seed=0):
    """
    Alternating Least Squares for Tensor Decomposition.
    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    We fix V and W, solve for U, then fix U and W, solve for V, etc.
    This is a non-convex optimization, so we use multiple random restarts.
    """
    np.random.seed(seed)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Solve for U
        # T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
        # For each a, we have a system.
        # Reshape T to (n*m, m*p*p*n) ? No, that's huge.
        # Better: For each a, T[a, :, :] is a matrix.
        # Let M_b_c = sum_r V[b,r] W[c,r] * U[a,r]
        # We want to find U[a, :] such that U[a, :] @ (V[:, r] * W[:, r])^T approx T[a, :, :]
        
        # Precompute "basis" matrices for V and W?
        # Actually, we can vectorize.
        # Let B[r, b, c] = V[b, r] * W[c, r]
        # Then T[a, b, c] = sum_r U[a, r] B[r, b, c]
        # This is a linear regression for each a.
        
        # Construct B: shape (rank, m*p, p*n)
        B = np.einsum('br,cr->rbc', V, W)
        
        # Solve for U
        # T[a, :, :] = U[a, :] @ B.reshape(rank, -1).T ?
        # B_flat[r, idx] where idx = b*(p*n) + c
        B_flat = B.reshape(rank, -1)
        
        # T_flat[a, idx]
        T_flat = T.reshape(n*m, -1)
        
        # Solve U @ B_flat.T = T_flat
        # U = T_flat @ B_flat.T @ inv(B_flat @ B_flat.T)
        try:
            # Use least squares
            U_new, _, _, _ = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Solve for V
        # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
        # Let C[r, a, c] = U[a, r] * W[c, r]
        C = np.einsum('ar,cr->rac', U, W)
        C_flat = C.reshape(rank, -1) # (rank, n*m * p*n)
        
        # T[b, a, c] ? No, T is indexed [a, b, c].
        # We need to rearrange T so that b is the row index.
        # T_reshaped[b, a, c] = T[a, b, c]
        T_reshaped = np.transpose(T, (1, 0, 2))
        T_flat_b = T_reshaped.reshape(m*p, -1)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(C_flat.T, T_flat_b.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Solve for W
        # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
        # Let D[r, a, b] = U[a, r] * V[b, r]
        D = np.einsum('ar,br->rab', U, V)
        D_flat = D.reshape(rank, -1) # (rank, n*m * m*p)
        
        # T[c, a, b] = T[a, b, c]
        T_reshaped_c = np.transpose(T, (2, 0, 1))
        T_flat_c = T_reshaped_c.reshape(p*n, -1)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(D_flat.T, T_flat_c.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        # Calculate error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
    return best_U, best_V, best_W, best_err

def _strassen_2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7) with trace convention W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A11-A10)(B00+B01): U[3]=1,U[2]=-1; V[0]=1,V[1]=1; C: W[0]=1,W[2]=-1,W[3]=1
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11): U[2]=1,U[3]=1; V[2]=1,V[3]=1; C: W[2]=1,W[3]=1
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00): U[0]=1,U[2]=1; V[0]=1; C: W[2]=1,W[3]=1
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11): U[3]=1,U[1]=1; V[3]=1; C: W[0]=1,W[2]=-1
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11): U[0]=1,U[2]=-1; V[1]=1,V[3]=1; C: W[0]=-1,W[2]=1
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00): U[1]=1,U[3]=1; V[2]=1,V[0]=-1; C: W[2]=-1,W[3]=-1
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10): U[0]=1,U[1]=1; V[0]=1,V[2]=-1; C: W[0]=1,W[2]=-1
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    return U, V, W

def _build_4x4x8_kronecker():
    """Build 4x4x8 via Kronecker: 2x2x2 Strassen -> 4x4x4 -> 4x4x8 (Rank 98)"""
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # 4x4x4 from 2x2x2
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for rb in range(7):
        for rs in range(7):
            col = rb * 7 + rs
            for ib in range(2):
                for jb in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            a = ib*8 + i0*4 + jb*2 + j0
                            ab = ib*2 + jb
                            as_idx = i0*2 + j0
                            U4[a][col] = U2[ab][rb] * U2[as_idx][rs]
                            
            for jb2 in range(2):
                for j02 in range(2):
                    for kb in range(2):
                        for k0 in range(2):
                            b = jb2*8 + j02*4 + kb*2 + k0
                            bb = jb2*2 + kb
                            bs = j02*2 + k0
                            V4[b][col] = V2[bb][rb] * V2[bs][rs]
                            
            for kb2 in range(2):
                for k02 in range(2):
                    for ib2 in range(2):
                        for i02 in range(2):
                            c = kb2*8 + k02*4 + ib2*2 + i02
                            cb = kb2*2 + ib2
                            cs = k02*2 + i02
                            W4[c][col] = W2[cb][rb] * W2[cs][rs]
                            
    # 4x4x8 from 4x4x4 (two copies)
    U8 = np.zeros((16, 98))
    V8 = np.zeros((32, 98))
    W8 = np.zeros((32, 98))
    
    for r in range(49):
        # First half
        U8[:, r] = U4[:, r]
        V8[:16, r] = V4[:, r]
        W8[:16, r] = W4[:, r]
        
        # Second half
        U8[:, r + 49] = U4[:, r]
        V8[16:, r + 49] = V4[:, r]
        W8[16:, r + 49] = W4[:, r]
        
    return U8, V8, W8

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    T = _generate_tensor(n, m, p)
    
    # Baseline: Kronecker Rank 98
    U_k, V_k, W_k = _build_4x4x8_kronecker()
    best_result = {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    best_rank = 98
    
    # Try to find lower rank via ALS
    # We know rank must be at least 64 (trivial lower bound? No, Strassen is 7 for 2x2x2, so 49 for 4x4x4. 98 for 4x4x8.)
    # Let's try ranks below 98.
    # Note: ALS is local. It might get stuck.
    
    ranks_to_try = [97, 96, 95, 94, 93, 92, 91, 90, 80, 70, 64]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
                
            U, V, W, err = _als_decompose(T, n, m, p, r, max_iter=200, tol=1e-5, seed=seed)
            
            if err < 1e-5:
                # Verify
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                    # Convert to list
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # If we found a valid lower rank, we can stop or try to go even lower
            # But ALS might be slow to converge for much lower ranks.
            # Let's break if we found something significantly better than 98?
            # Or just continue to try lower.
            pass
            
    # If we didn't find anything better, return the Kronecker result
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
