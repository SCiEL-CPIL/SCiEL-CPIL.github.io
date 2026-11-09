import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        target_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target_set.add((a, b, c))
                    if abs(T[a, b, c] - 1) > 1e-9:
                        return False
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def wftt_decomposition():
        """
        Winograd Fourier Transform Technique for 2x4 x 4x4 -> 2x4.
        
        The multiplication tensor T[i,j,k] = delta_{i,j} * delta_{j,k} (in the appropriate sense)
        can be transformed using the 4-point DFT on the j and k indices (both in Z4).
        
        In the frequency domain, this becomes four independent 2x2x2 tensor products,
        each solvable with rank-7 Strassen. Total rank = 4 * 7 = 28.
        
        Wait, that gives 28. Let me think more carefully.
        
        Actually, the WFTT approach: We want to compute C[i,k] = sum_j A[i,j] B[j,k].
        
        Using the DFT on the j and k indices:
        Let's define transforms on the 4-dimensional index space.
        
        The key structure: C[i,k] = sum_j A[i,j] B[j,k]
        
        This is a "convolution-like" operation. If we apply DFT_4 to both j and k:
        C_hat[i, omega] = sum_k C[i,k] * omega^k = sum_j A[i,j] * (sum_k B[j,k] omega^k)
        = sum_j A[i,j] * B_hat[j, omega]
        
        So for each frequency omega (4 values), we have:
        C_hat[:, omega] = A * B_hat[:, omega]
        
        where A is 2x4 and B_hat[:, omega] is 4x1.
        
        This gives us 4 independent matrix-vector multiplications of size 2x4 * 4x1 = 2x1.
        Each such multiplication has rank 2 (just 2 scalar multiplications), so total rank = 4*2 = 8?
        
        No wait, that's not quite right because we need to recover from the frequency domain.
        
        Let me reconsider. The proper WFTT approach:
        
        We have the tensor T with T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
        
        Actually, T[i,j,k] = 1 for all valid i,j,k (it's the standard matmul tensor).
        
        The WFTT insight: apply a transform to the j-index and k-index simultaneously.
        
        Let F be the 4x4 DFT matrix. Define:
        A' = A * F^{-1} (transform along j-axis) -> 2x4
        B' = F * B (transform along j-axis of B, i.e., row-axis) -> 4x4  
        
        Hmm, this is getting complex. Let me use a cleaner formulation.
        
        The standard WFTT for matrix multiplication M_{n x m} * M_{m x p}:
        If m is a power of 2, we can use the m-point DFT.
        
        For our case: n=2, m=4, p=4.
        
        Apply DFT_4 along the j-index of A (columns) and along the j-index of B (rows):
        A_hat[i, omega] = sum_j A[i,j] * omega^{-j}  (for omega in {0,1,2,3})
        B_hat[j, k] -> B_hat[omega, k] = sum_j B[j,k] * omega^{j}
        
        Then: C[i,k] = sum_j A[i,j] B[j,k]
        
        In frequency domain:
        C[i,k] = (1/4) sum_omega A_hat[i,omega] * B_hat[omega,k] * (some phase)
        
        Actually, the cleaner way: The DFT diagonalizes the "cyclic" structure.
        
        Let me think of it differently. The tensor is:
        T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{ij} * delta_{jk}... 
        
        No. T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
        The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
        
        The WFTT approach: We want to find a low-rank decomposition.
        
        Key insight: If we can write the tensor as a sum of rank-1 terms where the j and k indices are "decoupled" via a transform.
        
        Specifically, using the DFT:
        omega^4 = 1, omega = e^{2*pi*i/4} = i
        
        The DFT matrix F (4x4):
        F[a,b] = omega^{a*b} = i^{a*b}
        
        F = [[1, 1, 1, 1],
             [1, i, -1, -i],
             [1, -1, 1, -1],
             [1, -i, -1, i]]
        
        F^{-1} = (1/4) * F^* (conjugate transpose)
        
        Now, the WFTT decomposition:
        
        C[i,k] = sum_j A[i,j] B[j,k]
        
        Insert the DFT identity: delta_{j,k'} = (1/4) sum_omega omega^{omega*(j-k')}
        
        Actually, let's use: sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
        
        Write B[j,k] = (1/4) sum_{omega} B_hat[omega,k] * omega^{-j*omega}  ... no.
        
        B_hat[omega, k] = sum_j B[j,k] * omega^{j*omega}
        B[j,k] = (1/4) sum_{omega} B_hat[omega,k] * omega^{-j*omega}
        
        Similarly:
        A[i,j] = (1/4) sum_{omega'} A_hat[i,omega'] * omega'^{j*omega'}
        
        Hmm, this is getting complicated. Let me just try the direct approach.
        
        Actually, the simplest WFTT: 
        C[i,k] = sum_j A[i,j] B[j,k]
        
        Let's define for each omega in {0,1,2,3}:
        P_omega[i,k] = sum_j A[i,j] * omega^{j} * B[j,k] * omega^{-j}
        
        No, that's just A*B with phases.
        
        Let me try yet another approach. The rank of the 2x4x4 matmul tensor.
        
        Known results: The border rank of 2x4x4 matmul is at most 16 (by WFTT).
        Actually, I recall that for m x n x p matmul where one dimension is 2, there are specific bounds.
        
        Let me try a different structural approach: 
        Split the 4x4 B matrix into two 4x2 halves: B = [B_left | B_right]
        Split the 2x4 A matrix... hmm.
        
        Actually, let me try the approach where we use the fact that:
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        Group terms:
        C[i,k] = A[i,0]*B[0,k] + A[i,2]*B[2,k] + A[i,1]*B[1,k] + A[i,3]*B[3,k]
        
        This doesn't immediately help.
        
        Let me try the WFTT properly. The idea is:
        
        For the tensor T[i,j,k] (where i=0,1; j=0,1,2,3; k=0,1,2,3):
        
        Apply the 4-point DFT transform to the (j,k) pair. Specifically, define new indices.
        
        The WFTT says: if we have a tensor that is "separable" in a certain way under a group action, we can reduce the rank.
        
        For matrix multiplication, the relevant group is the cyclic group Z_4 acting on the j and k indices.
        
        The decomposition:
        T[i,j,k] = sum_{omega=0}^{3} (1/4) * omega^{j*omega} * omega^{-k*omega} * S[i, omega, j-k mod 4]
        
        Hmm, this is getting too abstract. Let me just try to construct a rank-16 decomposition directly.
        
        Alternative approach: Use the fact that 2x4x4 matmul can be done with rank 16.
        
        Here's a construction:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        Write j = 2*a + b where a,b in {0,1}
        Write k = 2*c + d where c,d in {0,1}
        
        C[i, 2c+d] = sum_{a,b} A[i, 2a+b] B[2a+b, 2c+d]
        
        This is like a 2x2 block structure but with the A and B blocks interleaved.
        
        For each (c,d) pair (4 output columns), we have:
        C[i, 2c+d] = sum_{a=0}^{1} sum_{b=0}^{1} A[i, 2a+b] B[2a+b, 2c+d]
        
        This is a 2x2 "convolution" in the (a,c) indices with b,d as free indices.
        
        For fixed b,d: sum_a A[i, 2a+b] B[2a+b, 2c+d]
        
        This is a 2x2 matmul (in the a,c indices) for each (b,d) pair.
        There are 4 such pairs: (b,d) in {(0,0),(0,1),(1,0),(1,1)}.
        
        Each 2x2 matmul has rank 7 (Strassen), so total rank = 4*7 = 28.
        Same as before.
        
        But wait, maybe we can do better by sharing some of the Strassen products.
        
        Actually, I think the minimal rank for 2x4x4 is indeed 16. Let me look for a construction.
        
        Here's an idea: Use 16 rank-1 terms.
        
        For each (omega, s) where omega in {0,1,2,3} and s in {0,1,2,3} (4*4=16 terms):
        
        U[i*m+j, (omega,s)] = omega^{j*omega} * delta_{s, something}
        
        Hmm, I'm not sure.
        
        Let me try a completely different approach: use the known result that the rank of 2xn x nxn matmul is at most 2n * ceil(log2(n)) or something.
        
        Actually, I recall that for 2x4x4, the rank is 16. Here's a construction using the WFTT:
        
        The key: the tensor T[i,j,k] can be written as:
        T[i,j,k] = (1/4) sum_{omega=0}^{3} omega^{j*omega} * omega^{-k*omega} * (4 * delta_{j,k})
        
        No, that's not right either.
        
        Let me try: T[i,j,k] = sum_j A[i,j] B[j,k]
        
        The "identity" in the tensor algebra: 
        delta_{j,k} = (1/4) sum_{omega} omega^{omega*(j-k)}
        
        But we don't have delta_{j,k} in our tensor; we have the full matmul.
        
        OK here's my final attempt at the WFTT:
        
        Define the 4x4 DFT matrix F with F[a,b] = i^{a*b} where i = sqrt(-1).
        
        Then F is unitary up to a factor: F * F^* = 4 * I.
        
        The WFTT decomposition:
        
        C = A * B
        
        Let A' = A * F^* / 2  (transform columns of A)
        Let B' = F / 2 * B  (transform rows of B)
        
        Then C = A' * B'  ... no, that's not correct.
        
        A * B = A * (F^* / 2) * (F / 2) * B = (A * F^* / 2) * (F / 2 * B)
        
        So C = A' * B' where A' = A * F^*/2 and B' = F/2 * B.
        
        But this doesn't reduce the rank; it just changes the representation.
        
        The rank reduction comes from the fact that in the frequency domain, the multiplication becomes "pointwise" in some sense.
        
        I think I'm overcomplicating this. Let me just try to verify if rank 16 is achievable by attempting a random search or a structured search.
        
        Actually, let me just return the rank-28 Strassen decomposition for now, but structure the code to allow for future improvements. The WFTT might not directly give rank 16 for this specific case.
        
        Wait, I just realized: the WFTT for 2x4x4 should give rank 4 * 7 = 28, same as block Strassen. The improvement to rank 16 would require a more sophisticated argument.
        
        Let me check: is there a known rank-16 decomposition for 2x4x4?
        
        After more thought, I believe the minimal rank is 16. Here's a construction:
        
        Use 16 rank-1 tensors indexed by (u, v) where u in {0,1,2,3} and v in {0,1,2,3}.
        
        For the u-th and v-th term:
        U[i*4+j, (u,v)] = i^{u*j} * i^{v*i}  ... no.
        
        Let me try:
        U[i*4+j, (u,v)] = i^{u*j}
        V[j*4+k, (u,v)] = i^{-u*j + v*k}  ... this doesn't work because V shouldn't depend on i.
        
        I think the correct WFTT construction is:
        
        T[i,j,k] = (1/4) sum_{omega=0}^{3} i^{omega*j} * i^{-omega*k} * 4*delta_{j,k}
        
        No, T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor, not a delta).
        
        I'm stuck on the theoretical construction. Let me just keep the rank-28 Strassen and add a pruning step to see if we can reduce it.
        
        Actually, let me try a different structural approach: use the fact that the 2x4x4 tensor has a specific symmetry and try to find a rank-16 decomposition via a different block decomposition.
        
        New idea: Split the 4x4 B into two 2x4 blocks vertically:
        B = [B_top; B_bottom] where B_top is 2x4 and B_bottom is 2x4.
        
        Then C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
        = sum_{j=0}^{1} A[i,j] B_top[j,k] + sum_{j=0}^{1} A[i,2+j] B_bottom[j,k]
        
        This is like two 2x2x4 multiplications added together.
        
        Each 2x2x4 matmul: C[i,k] = sum_j A[i,j] B[j,k] where A is 2x2 and B is 2x4.
        This can be seen as two independent 2x2x2 matmuls (for k=0,1 and k=2,3).
        Each 2x2x2 has rank 7, so 2x2x4 has rank 14.
        Total: 14 + 14 = 28. Still 28.
        
        I think 28 might actually be optimal for this case, or the improvement to 16 requires a very non-obvious construction.
        
        Let me just keep the rank-28 Strassen and focus on making the code robust.
        
        """
        # Standard Strassen factors for 2x2
        U2 = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ]
        
        V2 = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [1, 0, -1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        
        W2 = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, -1],
        ]
        
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        blocks = [
            (0, 0, 0, 0),
            (2, 2, 0, 0),
            (0, 0, 2, 2),
            (2, 2, 2, 2),
        ]
        
        col = 0
        for ac_off, br_off, bc_off, cc_off in blocks:
            for r in range(7):
                for li in range(4):
                    coeff = U2[r][li]
                    if coeff != 0:
                        local_row = li // 2
                        local_col = li % 2
                        global_a = local_row * 4 + (ac_off + local_col)
                        U[global_a][col] += coeff
                for lj in range(4):
                    coeff = V2[r][lj]
                    if coeff != 0:
                        local_row = lj // 2
                        local_col = lj % 2
                        global_b = (br_off + local_row) * 4 + (bc_off + local_col)
                        V[global_b][col] += coeff
                for lk in range(4):
                    coeff = W2[r][lk]
                    if coeff != 0:
                        local_row = lk // 2
                        local_col = lk % 2
                        global_c = (cc_off + local_col) * 2 + local_row
                        W[global_c][col] += coeff
                col += 1
                
        return {"U": U, "V": V, "W": W}

    def try_wftt_rank16():
        """
        Attempt a rank-16 WFTT-based decomposition.
        
        The idea: use the 4-point DFT to transform the problem into 4 independent
        2x2x2 multiplications in the frequency domain, but with shared structure
        that allows rank reduction.
        
        Specifically, we use the fact that the DFT matrix F (4x4) has entries
        F[a,b] = i^{a*b} where i = sqrt(-1).
        
        The WFTT decomposition of the 2x4x4 matmul tensor:
        
        For each frequency omega in {0,1,2,3}, define:
        A_omega[i, j] = A[i, j] (no transform on A for now)
        B_omega[j, k] = B[j, k] * i^{omega*j}
        
        Then: C_omega[i, k] = sum_j A[i,j] * B_omega[j,k] = sum_j A[i,j] B[j,k] i^{omega*j}
        
        And: C[i,k] = (1/4) sum_omega C_omega[i,k] * i^{-omega*k}
        
        This gives 4 independent 2x4x4 multiplications (each with a phase on B),
        which doesn't help directly.
        
        Alternative: transform both A and B.
        A_hat[i, omega] = sum_j A[i,j] i^{omega*j}
        B_hat[omega, k] = sum_j B[j,k] i^{-omega*j}
        
        Then: sum_j A[i,j] B[j,k] = (1/4) sum_omega A_hat[i,omega] B_hat[omega,k]
        
        So: C[i,k] = (1/4) sum_omega A_hat[i,omega] B_hat[omega,k]
        
        This means C = (1/4) * sum_omega A_hat[:,omega] * B_hat[omega,:]
        
        Each term A_hat[:,omega] * B_hat[omega,:] is a 2x4 outer product, which is rank 1
        as a matrix but rank 8 as a tensor (2*4*4/... no).
        
        Actually, A_hat[:,omega] is a 2x1 vector and B_hat[omega,:] is a 1x4 vector.
        Their outer product is a 2x4 matrix of rank 1.
        
        But we need the tensor decomposition, not matrix decomposition.
        
        The tensor T[i,j,k] = 1 for all i,j,k.
        
        Using the DFT:
        T[i,j,k] = (1/4) sum_omega i^{omega*j} * i^{-omega*k} * (4 * delta_{j,k})
        
        No, T[i,j,k] = 1, not delta_{j,k}.
        
        I think the WFTT doesn't directly apply here because the matmul tensor isn't a convolution.
        
        Let me try a different approach: assume rank 16 is possible and try to find it.
        
        For rank 16, we need 16 rank-1 terms.
        
        Let's parameterize:
        U[i*4+j, r] for i in {0,1}, j in {0,1,2,3}, r in {0,...,15}
        V[j*4+k, r] for j in {0,1,2,3}, k in {0,1,2,3}, r in {0,...,15}
        W[k*2+i, r] for k in {0,1,2,3}, i in {0,1}, r in {0,...,15}
        
        The constraint: sum_r U[i*4+j, r] * V[j'*4+k, r] * W[k*2+i', r] = delta_{j,j'} * delta_{i,i'}
        
        Wait, the target tensor is: T[a,b,c] = 1 if a = i*4+j, b = j*4+k, c = k*2+i for some i,j,k.
        So T[i*4+j, j*4+k, k*2+i] = 1 for all i,j,k.
        And T[a,b,c] = 0 otherwise.
        
        This is a very specific tensor. Let me see if I can find a rank-16 decomposition.
        
        After more thought, I believe the rank is actually 16. Here's a construction:
        
        Index the 16 terms by (u, v) where u, v in {0,1,2,3}.
        
        U[i*4+j, (u,v)] = i^{u*j} * i^{v*i}
        V[j*4+k, (u,v)] = i^{-u*j} * i^{v*k}  ... no, V shouldn't depend on i.
        
        Let me try:
        U[i*4+j, (u,v)] = i^{u*j}
        V[j*4+k, (u,v)] = i^{-u*j + v*k}  ... still depends on j which is OK for V.
        W[k*2+i, (u,v)] = i^{-v*k} * delta_{something}
        
        This is not working. Let me just keep rank 28.
        """
        return None

    best = schoolbook()
    best_rank = n * m * p

    if time.time() < deadline - 10:
        kron = wftt_decomposition()
        if verify(kron["U"], kron["V"], kron["W"], 28):
            best = kron
            best_rank = 28

    # Try WFTT rank-16 if time permits
    if time.time() < deadline - 5:
        wftt16 = try_wftt_rank16()
        if wftt16 is not None:
            if verify(wftt16["U"], wftt16["V"], wftt16["W"], 16):
                best = wftt16
                best_rank = 16

    return best
# EVOLVE-BLOCK-END
