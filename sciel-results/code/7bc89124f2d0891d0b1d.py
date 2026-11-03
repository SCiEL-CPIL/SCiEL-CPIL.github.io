import numpy as np
import time

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def baseline_66():
    U = np.zeros((n*m, 66), dtype=float)
    V = np.zeros((m*p, 66), dtype=float)
    W = np.zeros((p*n, 66), dtype=float)
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        
        U[A00, col]+=1; U[A11, col]+=1; V[B00, col]+=1; V[B11, col]+=1; W[C00, col]+=1; W[C11, col]+=1; col+=1
        U[A10, col]+=1; U[A11, col]+=1; V[B00, col]+=1; W[C10, col]+=1; W[C11, col]-=1; col+=1
        U[A00, col]+=1; V[B01, col]+=1; V[B11, col]-=1; W[C01, col]+=1; W[C11, col]+=1; col+=1
        U[A11, col]+=1; V[B10, col]+=1; V[B00, col]-=1; W[C00, col]+=1; W[C10, col]+=1; col+=1
        U[A00, col]+=1; U[A01, col]+=1; V[B11, col]+=1; W[C00, col]-=1; W[C01, col]+=1; col+=1
        U[A10, col]+=1; U[A00, col]-=1; V[B00, col]+=1; V[B01, col]+=1; W[C11, col]+=1; col+=1
        U[A01, col]+=1; U[A11, col]-=1; V[B10, col]+=1; V[B11, col]+=1; W[C00, col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        
        U[A0, col]+=1; V[B00, col]+=1; W[C0, col]+=1; col+=1
        U[A1, col]+=1; V[B10, col]+=1; W[C0, col]+=1; col+=1
        U[A0, col]+=1; V[B01, col]+=1; W[C1, col]+=1; col+=1
        U[A1, col]+=1; V[B11, col]+=1; W[C1, col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    
    return U[:, :col], V[:, :col], W[:, :col]

def verify(U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[Aidx(i,j)]
                v_vec = V[Bidx(j,k)]
                w_vec = W[Cidx(i,k)]
                s = np.dot(u_vec, v_vec * w_vec)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def try_merge(U, V, W, l1, l2):
    """Try to merge terms l1 and l2 into a single term.
    Returns (new_U, new_V, new_W) if successful, else None."""
    r = U.shape[1]
    
    # Compute the tensor contribution of term l1 + term l2
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = (U[Aidx(i,j), l1] * V[Bidx(j,k), l1] * W[Cidx(i,k), l1] +
                            U[Aidx(i,j), l2] * V[Bidx(j,k), l2] * W[Cidx(i,k), l2])
    
    # Check if T is rank 1: T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    # Strategy: find a non-zero entry, fix one factor, solve for others.
    
    # Find all non-zero entries
    nz = np.argwhere(np.abs(T) > 1e-12)
    if len(nz) == 0:
        # Both terms cancel to zero - just drop both
        idx = [i for i in range(r) if i != l1 and i != l2]
        return U[:, idx], V[:, idx], W[:, idx]
    
    # Try to factor T as A[i,j] * B[j,k] * C[i,k]
    # Fix A[0,0] = 1 if T[0,0,k] is non-zero for some k
    # Or use a more systematic approach.
    
    # Systematic approach:
    # Pick a reference entry (i0, j0, k0) where T[i0,j0,k0] != 0
    # Set A[i0,j0] = 1
    # Then B[j0,k0] * C[i0,k0] = T[i0,j0,k0]
    # For any (i, j0, k): A[i,j0] * B[j0,k] * C[i,k] = T[i,j0,k]
    # For any (i0, j, k0): A[i0,j] * B[j,k0] * C[i0,k0] = T[i0,j,k0]
    
    # This is complex. Let's use a simpler check:
    # T is rank 1 iff for all i,i',j,j',k,k':
    # T[i,j,k]*T[i',j',k'] = T[i,j',k]*T[i',j,k'] (when denominators non-zero)
    # More practically: unfold T along different axes and check rank.
    
    # Unfold 1: (i*m+j) x k -> should be rank 1 if T is rank 1? No.
    # T[i,j,k] = A[i,j]*B[j,k]*C[i,k]
    # Unfold (i,j) x k: M1[i*m+j, k] = A[i,j] * (B[j,k]*C[i,k]) - not separable in general
    # Unfold i x (j,k): M2[i, j*p+k] = C[i,k] * A[i,j] * B[j,k] - not separable
    # 
    # Actually, the correct test: T is a rank-1 tensor in the sense of the
    # matrix multiplication tensor decomposition iff it can be written as
    # A[i,j] * B[j,k] * C[i,k].
    #
    # A practical test: if T has non-zero entries, pick (i0,j0,k0) with T[i0,j0,k0]!=0.
    # Set A[i0,j0] = 1.
    # Then for all (i,j0,k): A[i,j0] * B[j0,k] * C[i,k] = T[i,j0,k]
    # And for all (i0,j,k0): A[i0,j] * B[j,k0] * C[i0,k0] = T[i0,j,k0]
    # And for all (i,j,k): A[i,j] * B[j,k] * C[i,k] = T[i,j,k]
    
    # Let's try: assume there exist A, B, C.
    # Fix A[i0,j0] = 1.
    # From T[i0,j0,k] = B[j0,k] * C[i0,k], we have B[j0,k]/C[i0,k] = T[i0,j0,k]/T[i0,j0,k0] * C[i0,k0]/C[i0,k]... 
    # This is getting circular.
    
    # Alternative: use the fact that if T[i,j,k] = A[i,j]B[j,k]C[i,k], then:
    # T[i,j,k] * T[i',j',k'] = T[i,j',k] * T[i',j,k'] for all i,i',j,j',k,k'
    # (this is the rank-1 condition for the 3-way tensor in this specific format)
    
    # Check this condition on a sample of entries
    for i in range(n):
        for i2 in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            lhs = T[i,j,k] * T[i2,j2,k2]
                            rhs = T[i,j2,k] * T[i2,j,k2]
                            if abs(lhs - rhs) > 1e-9 * max(1, abs(lhs), abs(rhs)):
                                return None
    
    # T is rank 1 in the required sense. Now extract factors.
    # Find a non-zero entry
    i0, j0, k0 = nz[0]
    val = T[i0, j0, k0]
    
    A = np.zeros((n, m))
    B = np.zeros((m, p))
    C = np.zeros((p, n))
    
    # Set A[i0, j0] = 1
    A[i0, j0] = 1.0
    # Then B[j0, k0] * C[i0, k0] = val
    # We have freedom in scaling. Let's set C[i0, k0] = 1, B[j0, k0] = val
    C[i0, k0] = 1.0
    B[j0, k0] = val
    
    # Now determine all A[i, j0] from T[i, j0, k0] = A[i,j0] * B[j0,k0] * C[i,k0]
    # We know B[j0,k0] = val, but C[i,k0] is unknown for i != i0.
    # This is underdetermined. Let's use a different strategy.
    
    # Better: use the relations:
    # T[i, j0, k] = A[i,j0] * B[j0,k] * C[i,k]
    # T[i0, j, k0] = A[i0,j] * B[j,k0] * C[i0,k0]
    # T[i, j, k] = A[i,j] * B[j,k] * C[i,k]
    
    # Let's fix C[i0, k] for all k from T[i0, j0, k] = A[i0,j0]*B[j0,k]*C[i0,k] = 1*B[j0,k]*C[i0,k]
    # And T[i0, j, k0] = A[i0,j]*B[j,k0]*C[i0,k0]
    
    # This is still complex. Let me use a numerical approach:
    # Since we verified rank-1, we can solve for factors by picking a basis.
    
    # Simplest extraction:
    # Set A[i,j] = T[i,j,k0] / (B[j,k0]*C[i,k0]) for a fixed k0 where entries are non-zero.
    # But B and C are unknown.
    
    # Let's try: set C[i,k] = T[i,j0,k] / T[i0,j0,k0] * val for all i,k (using j0,j0 relation)
    # No, that's not right either.
    
    # Practical extraction:
    # We know T[i,j,k] = A[i,j]B[j,k]C[i,k]
    # Fix A[i0,j0] = 1, C[i0,k0] = 1, so B[j0,k0] = T[i0,j0,k0] = val
    # For all i: T[i,j0,k0] = A[i,j0]*B[j0,k0]*C[i,k0] = A[i,j0]*val*C[i,k0]
    # For all k: T[i0,j0,k] = A[i0,j0]*B[j0,k]*C[i0,k] = 1*B[j0,k]*C[i0,k]
    # For all j: T[i0,j,k0] = A[i0,j]*B[j,k0]*C[i0,k0] = A[i0,j]*B[j,k0]*1
    
    # Let's set B[j0,k] = T[i0,j0,k] for all k (since A[i0,j0]=1, C[i0,k]=1 for k=k0, but not for other k)
    # This doesn't work cleanly.
    
    # Let's just use a least-squares approach to find A, B, C given T.
    # Actually, since we've verified the rank-1 condition, let's just construct factors:
    
    # For each (i,j), A[i,j] can be determined from T[i,j,k]/(B[j,k]*C[i,k])
    # Let's fix C[i,k] = 1 for all i,k, then A[i,j]*B[j,k] = T[i,j,k]
    # This reduces to: for each j, the matrix T[:,j,:] (n x p) must be rank 1.
    # Check: for each j, T[:,j,:] should be rank 1.
    
    valid = True
    for j in range(m):
        Mj = T[:, j, :]  # n x p
        if Mj.size > 0:
            rank_j = np.linalg.matrix_rank(Mj)
            if rank_j > 1:
                valid = False
                break
    
    if not valid:
        return None
    
    # If valid, set C[i,k] = 1 for all i,k
    C = np.ones((p, n))
    # Then A[i,j]*B[j,k] = T[i,j,k]
    # For each j, T[:,j,:] = A[:,j] * B[j,:]^T (outer product)
    A = np.zeros((n, m))
    B = np.zeros((m, p))
    
    for j in range(m):
        Mj = T[:, j, :]  # n x p
        if np.all(np.abs(Mj) < 1e-12):
            continue
        # Find non-zero entry
        nz_j = np.argwhere(np.abs(Mj) > 1e-12)
        if len(nz_j) == 0:
            continue
        ii, kk = nz_j[0]
        val_j = Mj[ii, kk]
        # Set A[ii,j] = 1, B[j,kk] = val_j
        A[ii, j] = 1.0
        B[j, kk] = val_j
        # Fill in rest
        for i2 in range(n):
            if i2 != ii and abs(Mj[i2, kk]) > 1e-12:
                A[i2, j] = Mj[i2, kk] / val_j
        for k2 in range(p):
            if k2 != kk and abs(Mj[ii, k2]) > 1e-12:
                B[j, k2] = Mj[ii, k2] / 1.0  # A[ii,j] = 1
        
        # Verify consistency
        for i2 in range(n):
            for k2 in range(p):
                if abs(A[i2,j] * B[j,k2] - Mj[i2,k2]) > 1e-9 * max(1, abs(Mj[i2,k2])):
                    return None
    
    # Build flat arrays
    A_flat = np.zeros(n*m)
    B_flat = np.zeros(m*p)
    C_flat = np.zeros(p*n)
    for i in range(n):
        for j in range(m):
            A_flat[Aidx(i,j)] = A[i,j]
    for j in range(m):
        for k in range(p):
            B_flat[Bidx(j,k)] = B[j,k]
    for k in range(p):
        for i in range(n):
            C_flat[Cidx(i,k)] = C[i,k]
    
    # Build new U, V, W with merged term
    new_U = np.zeros((n*m, r-1))
    new_V = np.zeros((m*p, r-1))
    new_W = np.zeros((p*n, r-1))
    
    idx = 0
    for l in range(r):
        if l == l1 or l == l2:
            continue
        new_U[:, idx] = U[:, l]
        new_V[:, idx] = V[:, l]
        new_W[:, idx] = W[:, l]
        idx += 1
    
    new_U[:, idx] = A_flat
    new_V[:, idx] = B_flat
    new_W[:, idx] = C_flat
    
    # Verify
    if verify(new_U, new_V, new_W):
        return new_U, new_V, new_W
    return None

def local_search(U, V, W, deadline):
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    r = best_U.shape[1]
    
    while time.time() < deadline - 3:
        improved = False
        
        # Try to drop zero terms
        for l in range(r):
            if not (np.any(best_U[:, l]) and np.any(best_V[:, l]) and np.any(best_W[:, l])):
                idx = [i for i in range(r) if i != l]
                best_U = best_U[:, idx]
                best_V = best_V[:, idx]
                best_W = best_W[:, idx]
                r -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try to merge pairs
        merged = False
        for l1 in range(r):
            if time.time() > deadline - 3:
                break
            for l2 in range(l1+1, r):
                if time.time() > deadline - 3:
                    break
                result = try_merge(best_U, best_V, best_W, l1, l2)
                if result is not None:
                    best_U, best_V, best_W = result
                    r -= 1
                    merged = True
                    break
            if merged:
                break
        
        if not improved and not merged:
            break
    
    return r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 145
    U, V, W = baseline_66()
    r = U.shape[1]
    
    r, U, V, W = local_search(U, V, W, deadline)
    
    if not verify(U, V, W):
        U, V, W = baseline_66()
        r = 66
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
