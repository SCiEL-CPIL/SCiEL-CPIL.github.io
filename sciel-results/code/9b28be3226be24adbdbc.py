import time
import numpy as np

N, M, P = 2, 3, 10
START = time.time()

def _elapsed():
    return time.time() - START

def _verify(U, V, W, tol=1e-9):
    """Verify decomposition using numpy."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Efficient verification
    # For each r, compute outer products
    # We want sum_r U[:,r] V[:,r] W[:,r] to match the target tensor
    # Target tensor T[i,j,k] = 1 for all i in [0,N), j in [0,M), k in [0,P)
    # But wait, the tensor definition is such that C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entries T[i,j,k] are 1 if the index matches the multiplication pattern.
    # Actually, the tensor is defined by the bilinear map.
    # T[i,j,k] = 1 if (i,j,k) is a valid index for the multiplication?
    # No, the tensor M_{n,m,p} is a tensor of shape (n, m, p) such that
    # C = A B <=> c_k = sum_{i,j} M_{i,j,k} a_i b_j.
    # Here indices are 0-based.
    # For matrix multiplication, M_{i,j,k} = 1 if j is the column of A and row of B.
    # Specifically, T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # So T_{i,j,k} = 1 for all valid i, j, k.
    # Let's check the standard convention.
    # The multiplication tensor has entries 1 at (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # So yes, T[i,j,k] = 1 for all i in [0,N), j in [0,M), k in [0,P).
    
    # Let's construct the target tensor
    target = np.ones((N, M, P))
    
    # Compute the reconstructed tensor
    # Reconstructed[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: The task says W[k*n+i, r].
    # So W index is k*N + i.
    # U index is i*M + j.
    # V index is j*P + k.
    
    rec = np.zeros((N, M, P))
    for r in range(rank):
        u_r = U[:, r].reshape(N, M)
        v_r = V[:, r].reshape(M, P)
        w_r = W[:, r].reshape(P, N) # W is indexed by k*N+i, so reshape to (P, N) gives w[k, i]
        
        # We need to add u_r[i,j] * v_r[j,k] * w_r[k,i] to rec[i,j,k]
        # This is a sum over j for fixed i,k? No, we are summing over r.
        # For each r, we add a rank-1 tensor (in the 3-mode sense) but with specific index couplings.
        # Actually, it's just element-wise multiplication of reshaped vectors?
        # rec[i,j,k] += U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
        
        # We can use einsum for speed, but simple loops are fine for small sizes.
        # Or just broadcast.
        # u_r shape (N, M, 1)
        # v_r shape (1, M, P)
        # w_r shape (P, 1, N) -> need to align to (N, M, P)
        # w_r[k, i] -> index (i, j, k) needs w[k, i].
        # So w part is w_r.T? No.
        # Let's just loop i, j, k? 2*3*10 = 60 entries. Rank ~40. 2400 ops. Negligible.
        
        for i in range(N):
            for j in range(M):
                u_val = U[i * M + j, r]
                if u_val == 0: continue
                for k in range(P):
                    v_val = V[j * P + k, r]
                    if v_val == 0: continue
                    w_val = W[k * N + i, r]
                    if w_val == 0: continue
                    rec[i, j, k] += u_val * v_val * w_val
                    
    if not np.allclose(rec, target, atol=tol):
        return False
    return True

def construct():
    # Rank 40 solution based on Strassen blocks
    # 5 blocks of 2x2x2 Strassen (rank 7) + 5 blocks of 1x1x1 (rank 1 for j=2)
    # Wait, the j=2 part is A[:,2] * B[2,:].
    # Tensor part for j=2: T[i, 2, k] = 1 for all i, k.
    # This is a 2x1x10 tensor? No, for fixed block of k's.
    # In the current design, we split k into 5 blocks of 2.
    # For block b, k in {2b, 2b+1}.
    # We need to satisfy T[i, 2, k] = 1 for i in {0,1}, k in {2b, 2b+1}.
    # This sub-tensor is 2x1x2.
    # Can we decompose 2x1x2 tensor in rank 1?
    # T[i, j=2, k] = 1.
    # We need sum_r U[i, r] V[2, r] W[k, r] = 1.
    # If we use 1 term: U[i, 1] = 1, V[2, 1] = 1, W[k, 1] = 1.
    # Then 1*1*1 = 1.
    # Yes, rank 1 suffices for each 2x1x2 block.
    # So total rank = 5 * (7 + 1) = 40.
    
    rank = 40
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    col = 0
    # Indices
    # U: i*M + j
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        # V indices: j*P + k
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        b20, b21 = 20 + k0, 20 + k1
        
        # W indices: k*N + i
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen 2x2x2 for j in {0,1}, i in {0,1}, k in {k0, k1}
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6 = (A10+A00)(B00+B01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        # Direct term for j=2
        # U[i, r] = 1 for i in {0,1}
        # V[2, r] = 1
        # W[k, r] = 1 for k in {k0, k1}
        U[a00, col] += 1; U[a10, col] += 1
        V[b20, col] += 1 # This is V[20+k0, col] which is V[2, k0]
        W[c00, col] += 1; W[c10, col] += 1 # W[k0, 0] and W[k0, 1]
        W[c01, col] += 1; W[c11, col] += 1 # W[k1, 0] and W[k1, 1]
        col += 1

    # Verify
    if not _verify(U, V, W):
        # Fallback to rank 55 schoolbook/strassen mix if something is wrong
        # This is just to ensure validity, though rank 40 should work.
        U55, V55, W55 = _build_fallback_rank55()
        return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_fallback_rank55():
    """Fallback construction with rank 55."""
    rank = 55
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen 7 terms
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        # Schoolbook for j=2 (4 terms)
        # A02 B20 -> C00
        U[a02, col] += 1; V[20+k0, col] += 1; W[c00, col] += 1
        col += 1
        # A02 B21 -> C01
        U[a02, col] += 1; V[20+k1, col] += 1; W[c01, col] += 1
        col += 1
        # A12 B20 -> C10
        U[a12, col] += 1; V[20+k0, col] += 1; W[c10, col] += 1
        col += 1
        # A12 B21 -> C11
        U[a12, col] += 1; V[20+k1, col] += 1; W[c11, col] += 1
        col += 1

    return U, V, W

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    print(f"Rank: {U.shape[1]}")
    print("Verified:", _verify(U, V, W))
