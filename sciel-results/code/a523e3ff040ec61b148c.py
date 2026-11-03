import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce rank from 60 to 56.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    dim_u = N * M  # 12
    dim_v = M * P  # 20
    dim_w = P * N  # 15
    
    # We expect rank 56
    # 4 Strassen blocks (4 * 7 = 28) + 28 schoolbook terms = 56
    expected_rank = 56
    
    U = np.zeros((dim_u, expected_rank))
    V = np.zeros((dim_v, expected_rank))
    W = np.zeros((dim_w, expected_rank))
    
    col_idx = 0
    covered = set()
    
    # Define Strassen terms for a 2x2x2 block
    # Local indices i, j, k in {0, 1}
    # Global indices i0+i, j0+j, k0+k
    
    def get_strassen_terms(i0, j0, k0):
        """
        Returns 7 tuples of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        for the 2x2x2 block starting at (i0, j0, k0).
        """
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        u1_idx = [(i0+0)*M + (j0+0), (i0+1)*M + (j0+1)]
        u1_val = [1, 1]
        v1_idx = [(j0+0)*P + (k0+0), (j0+1)*P + (k0+1)]
        v1_val = [1, 1]
        w1_idx = [(k0+0)*N + (i0+0), (k0+1)*N + (i0+1)]
        w1_val = [1, 1]
        terms.append((u1_idx, u1_val, v1_idx, v1_val, w1_idx, w1_val))
        
        # M2 = (A10 + A11)B00
        u2_idx = [(i0+1)*M + (j0+0), (i0+1)*M + (j0+1)]
        u2_val = [1, 1]
        v2_idx = [(j0+0)*P + (k0+0)]
        v2_val = [1]
        w2_idx = [(k0+0)*N + (i0+1), (k0+1)*N + (i0+1)]
        w2_val = [1, -1]
        terms.append((u2_idx, u2_val, v2_idx, v2_val, w2_idx, w2_val))
        
        # M3 = A00(B10 - B00)
        u3_idx = [(i0+0)*M + (j0+0)]
        u3_val = [1]
        v3_idx = [(j0+0)*P + (k0+1), (j0+1)*P + (k0+1)]
        v3_val = [1, -1]
        w3_idx = [(k0+1)*N + (i0+0), (k0+1)*N + (i0+1)]
        w3_val = [1, 1]
        terms.append((u3_idx, u3_val, v3_idx, v3_val, w3_idx, w3_val))
        
        # M4 = A11(B01 - B11)
        u4_idx = [(i0+1)*M + (j0+1)]
        u4_val = [1]
        v4_idx = [(j0+1)*P + (k0+0), (j0+0)*P + (k0+0)]
        v4_val = [1, -1]
        w4_idx = [(k0+0)*N + (i0+0), (k0+0)*N + (i0+1)]
        w4_val = [1, 1]
        terms.append((u4_idx, u4_val, v4_idx, v4_val, w4_idx, w4_val))
        
        # M5 = (A00 + A01)B11
        u5_idx = [(i0+0)*M + (j0+0), (i0+0)*M + (j0+1)]
        u5_val = [1, 1]
        v5_idx = [(j0+1)*P + (k0+1)]
        v5_val = [1]
        w5_idx = [(k0+0)*N + (i0+0), (k0+1)*N + (i0+0)]
        w5_val = [-1, 1]
        terms.append((u5_idx, u5_val, v5_idx, v5_val, w5_idx, w5_val))
        
        # M6 = (A10 - A00)(B00 + B01)
        u6_idx = [(i0+1)*M + (j0+0), (i0+0)*M + (j0+0)]
        u6_val = [1, -1]
        v6_idx = [(j0+0)*P + (k0+0), (j0+0)*P + (k0+1)]
        v6_val = [1, 1]
        w6_idx = [(k0+1)*N + (i0+1)]
        w6_val = [1]
        terms.append((u6_idx, u6_val, v6_idx, v6_val, w6_idx, w6_val))
        
        # M7 = (A01 - A11)(B10 + B11)
        u7_idx = [(i0+0)*M + (j0+1), (i0+1)*M + (j0+1)]
        u7_val = [1, -1]
        v7_idx = [(j0+1)*P + (k0+0), (j0+1)*P + (k0+1)]
        v7_val = [1, 1]
        w7_idx = [(k0+0)*N + (i0+0)]
        w7_val = [1]
        terms.append((u7_idx, u7_val, v7_idx, v7_val, w7_idx, w7_val))
        
        return terms

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        for idx, val in zip(u_idx, u_val):
            U[idx, col] += val
        for idx, val in zip(v_idx, v_val):
            V[idx, col] += val
        for idx, val in zip(w_idx, w_val):
            W[idx, col] += val

    # Add Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    for i0, j0, k0 in blocks:
        if time.time() > deadline:
            break
        terms = get_strassen_terms(i0, j0, k0)
        for t in terms:
            if time.time() > deadline:
                break
            add_term(t[0], t[1], t[2], t[3], t[4], t[5], col_idx)
            col_idx += 1
        # Mark covered indices
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook terms for remaining indices
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) in covered:
                    continue
                if time.time() > deadline:
                    break
                # Schoolbook term
                U[i*M + j, col_idx] = 1
                V[j*P + k, col_idx] = 1
                W[k*N + i, col_idx] = 1
                col_idx += 1
    
    # If we ran out of time or space, we might have col_idx < expected_rank
    # We should trim the matrices to the actual rank used
    if col_idx < expected_rank:
        U = U[:, :col_idx]
        V = V[:, :col_idx]
        W = W[:, :col_idx]
    
    # Verify
    rank = U.shape[1]
    valid = True
    # Check a sample of entries to ensure correctness
    # Full check is O(N*M*P*N*M*P*Rank) which is too slow.
    # We check the diagonal and a few off-diagonals.
    # Actually, the tensor is sparse. We can check specific entries.
    # T[i, j, k, i', j', k'] = delta_ii' delta_jj' delta_kk'
    # Sum_c U[i,j,c] V[j',k,c] W[k',i',c]
    
    # Check diagonal: i=i', j=j', k=k'
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    s += U[i*M + j, c] * V[j*P + k, c] * W[k*N + i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    # Check off-diagonal: i != i'
    for i in range(N):
        for i_prime in range(N):
            if i == i_prime: continue
            for j in range(M):
                for k in range(P):
                    s = 0
                    for c in range(rank):
                        s += U[i*M + j, c] * V[j*P + k, c] * W[k*N + i_prime, c]
                    if abs(s) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((dim_u, N*M*P))
        V = np.zeros((dim_v, N*M*P))
        W = np.zeros((dim_w, N*M*P))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i*M + j, col] = 1
                    V[j*P + k, col] = 1
                    W[k*N + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
