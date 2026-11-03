"""Combined: block-level Strassen (inspiration structure) + verification & deadline guard (refinement)."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 4, 4, 6
    rank = 77

    U = [[0] * rank for _ in range(n * m)]   # 16 x 77
    V = [[0] * rank for _ in range(m * p)]   # 24 x 77
    W = [[0] * rank for _ in range(p * n)]   # 24 x 77

    # Outer Strassen: partition A(4x4) into 2x2 blocks of 2x2,
    # B(4x6) into 2x2 blocks of 2x3, C(4x6) into 2x2 blocks of 2x3.
    # Block ids: 0=00, 1=01, 2=10, 3=11.
    # Each step: (A_coeffs, B_coeffs, C_coeffs) as list of (block_id, coeff).
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])),
    ]

    # Global index mappers.
    # A block (2x2): local (r,s) -> global (I,J). Block id b: I=(b//2)*2+r, J=(b%2)*2+s
    def gA(b, r, s):
        return (b // 2) * 2 + r, (b % 2) * 2 + s

    # B block (2x3): local (s,k) -> global (J,K). Block id b: J=(b//2)*2+s, K=(b%2)*3+k
    def gB(b, s, k):
        return (b // 2) * 2 + s, (b % 2) * 3 + k

    # C block (2x3): local (r,k) -> global (I,K). Block id b: I=(b//2)*2+r, K=(b%2)*3+k
    def gC(b, r, k):
        return (b // 2) * 2 + r, (b % 2) * 3 + k

    # Inner decomposition for 2x2 * 2x3 -> 2x3.
    # Part A: Strassen for first 2 columns (2x2 * 2x2), 7 terms.
    #   local A(r,s), B(s,k) with r,s,k in {0,1}, C(r,k).
    inner_strassen = [
        (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])),
        (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])),
        (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])),
        (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])),
        (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1), (0, 0, -1)])),
        (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])),
        (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])),
    ]

    # Part B: schoolbook for 3rd column (2x2 * 2x1 -> 2x1), 4 terms.
    #   local A(r,s), B(s,col2) -> C(r,col2). The single column maps to local k=2 in the 2x3 block.
    inner_schoolbook = [
        # A(r,s) * B(s, col2) -> C(r, col2)
        (([(0, 0, 1)], [(0, 2, 1)], [(0, 2, 1)])),
        (([(0, 1, 1)], [(1, 2, 1)], [(0, 2, 1)])),
        (([(1, 0, 1)], [(0, 2, 1)], [(1, 2, 1)])),
        (([(1, 1, 1)], [(1, 2, 1)], [(1, 2, 1)])),
    ]

    def add_term(rank_idx, A_terms, B_terms, C_terms, A_outer, B_outer, C_outer):
        for (r, s, ca) in A_terms:
            for (b, cb) in A_outer:
                I, J = gA(b, r, s)
                U[I * m + J][rank_idx] += ca * cb
        for (s, k, cb) in B_terms:
            for (b, cb2) in B_outer:
                J, K = gB(b, s, k)
                V[J * p + K][rank_idx] += cb * cb2
        for (r, k, cc) in C_terms:
            for (b, cb2) in C_outer:
                I, K = gC(b, r, k)
                W[K * n + I][rank_idx] += cc * cb2

    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = i * 11
        for j, (Ai, Bi, Ci) in enumerate(inner_strassen):
            add_term(base + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
        for j, (Ai, Bi, Ci) in enumerate(inner_schoolbook):
            add_term(base + 7 + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)

        if time.time() > deadline - 20:
            break

    # Verification: rebuild tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # and compare to the standard matmul tensor T[i,j,k] = A[i,j]*B[j,k] = delta structure.
    # For matmul tensor, T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0? No.
    # The matmul tensor T_{ijk} = A_{ij} B_{jk} C^{ik} contraction: T[i,j,k] = 1 always? 
    # Actually the tensor for matmul is T[i,j,k] = 1 for all i,j,k (the structure tensor).
    # Verify: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal 1 for all i,j,k.
    import numpy as np
    if time.time() < deadline - 5:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)
        # Build T[i,j,k] efficiently using tensor contractions
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape for efficient computation
        # U reshaped to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = Um.reshape(n, m, rank)
        V3 = Vm.reshape(m, p, rank)
        W3 = Wm.reshape(p, n, rank)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: sum over r and j for fixed i,k? No, we need all (i,j,k)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a 3-index contraction. Compute via:
        # First: UV[i,j,k,r] = U3[i,j,r] * V3[j,k,r] -- but this is expensive
        # Better: for each r, compute outer product and contract
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = Um[:, r]  # (n*m,)
            v_r = Vm[:, r]  # (m*p,)
            w_r = Wm[:, r]  # (p*n,)
            # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
            # For each (i,j): u_r[i*m+j], then for each k: v_r[j*p+k] * w_r[k*n+i]
            u_reshaped = u_r.reshape(n, m)
            v_reshaped = v_r.reshape(m, p)
            w_reshaped = w_r.reshape(p, n)
            # T_r[i,j,k] = u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            # = u_reshaped[i,j] * (v_reshaped[j,:] * w_reshaped[:,i])
            # For fixed (i,j): u_reshaped[i,j] * v_reshaped[j,:] * w_reshaped[:,i]
            # v_reshaped[j,:] is (p,), w_reshaped[:,i] is (p,), element-wise product is (p,)
            # So T_r[i,j,:] = u_reshaped[i,j] * (v_reshaped[j,:] * w_reshaped[:,i])
            # Vectorized: for all i,j:
            # vw[j,k,i] = v_reshaped[j,k] * w_reshaped[k,i]
            # Then T_r[i,j,k] = u_reshaped[i,j] * vw[j,k,i]
            vw = v_reshaped[:, :, None] * w_reshaped[None, :, :]  # (m, p, n) -> vw[j,k,i]
            # T_r[i,j,k] = u_reshaped[i,j] * vw[j,k,i]
            # u_reshaped[i,j] * vw[j,k,i] -> sum over j? No, this is per (i,j,k)
            # T_r[i,j,k] = u_reshaped[i,j] * vw[j,k,i]
            # = (u_reshaped[:, :, None] * vw[None, :, :].transpose(1,0,2))[:, :, :]
            # Let me just do it element-wise with broadcasting
            # u_reshaped[i,j] * vw[j,k,i]: 
            # u_reshaped shape (n,m), vw shape (m,p,n)
            # We want result (n,m,p) where result[i,j,k] = u_reshaped[i,j] * vw[j,k,i]
            # vw[j,k,i] can be transposed to vw_t[i,j,k] = vw[j,k,i]
            vw_t = vw.transpose(2, 0, 1)  # (n, m, p), vw_t[i,j,k] = vw[j,k,i]
            T_r = u_reshaped[:, :, None] * vw_t  # (n, m, p)
            T += T_r
        # The matmul structure tensor is all ones.
        ok = np.allclose(T, 1.0, atol=1e-9)
        if not ok:
            # Debug: find where it fails
            # Fallback: schoolbook (rank 96) is guaranteed correct.
            rank2 = n * m * p
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i*m+j][r] = 1
                        V2[j*p+k][r] = 1
                        W2[k*n+i][r] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
