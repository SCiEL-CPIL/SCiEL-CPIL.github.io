import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6

    rank = 24
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Systematic rank-24 construction:
    # For each j in range(m) and k in range(p), we need terms that map
    # A[i,j] * B[j,k] -> C[i,k] for i=0,1.
    # 
    # Key insight: Use 2 terms per (j,k) pair, one for each i value.
    # Term for (j,k,i): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives exactly the contribution A[i,j]*B[j,k] to C[i,k].
    #
    # Total rank = m * p * n = 4 * 6 * 2 = 48? No, that's the trivial decomposition.
    # 
    # Better approach: For each (j,k), use 2 terms (one per i):
    # Actually, let's use a smarter construction.
    #
    # For each j in 0..3 and k in 0..5:
    #   We need C[0,k] += A[0,j]*B[j,k] and C[1,k] += A[1,j]*B[j,k]
    #   Term 1: U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
    #   Term 2: U[1*m+j, r+1] = 1, V[j*p+k, r+1] = 1, W[k*n+1, r+1] = 1
    #
    # This gives rank = 2*m*p = 2*4*6 = 48. Still too high.
    #
    # Let me think more carefully. The tensor T[i,j,k] = 1 for all i,j,k.
    # We need to decompose this tensor.
    #
    # Actually, the correct approach:
    # For each j, we can factor out the B[j,k] dependence.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Let's use a block structure. For each j, define:
    #   u_j[i] = A[i,j] for i=0,1
    #   v_j[k] = B[j,k] for k=0..5
    #   C[:,k] += u_j * v_j[k]
    #
    # This is an outer product structure. For each j, we have a rank-1 update
    # to C using u_j and v_j.
    #
    # In tensor form, for each j, we need terms that compute:
    #   sum_i sum_k A[i,j] B[j,k] C[k,i]
    #   = (sum_i A[i,j] W[k*n+i]) * B[j,k] ... no, this doesn't work directly.
    #
    # Let me use the standard bilinear decomposition approach:
    # For each j, create m_j terms where m_j is the "rank" for that j.
    #
    # Actually, the simplest correct decomposition:
    # For each (i, j, k) with i in {0,1}, j in {0,1,2,3}, k in {0,...,5}:
    #   T[i,j,k] = 1
    #
    # Trivial decomposition has rank = n*m*p = 2*4*6 = 48.
    #
    # Can we do better? Yes, by exploiting structure.
    #
    # Key observation: For fixed j and k, the terms for i=0 and i=1 are:
    #   T[0,j,k] = 1 and T[1,j,k] = 1
    #
    # We can combine these into 2 terms per (j,k) pair:
    #   Term A: U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
    #   Term B: U[1*m+j, r+1] = 1, V[j*p+k, r+1] = 1, W[k*n+1, r+1] = 1
    #
    # This gives rank = 2 * m * p = 48. Same as trivial.
    #
    # To reduce rank, we need to find linear dependencies.
    #
    # Alternative: Use the fact that C[0,k] and C[1,k] share the same B[j,k].
    # For each j, define a 2x1 vector u_j = [A[0,j], A[1,j]]^T
    # and a 1x6 vector v_j = [B[j,0], ..., B[j,5]]
    # Then C += u_j * v_j (outer product)
    #
    # In tensor decomposition form, for each j, we can use 2 terms:
    #   Term 1: U[0*m+j, r] = 1, V[j*p+0, r] = 1, W[0*n+0, r] = 1
    #            U[0*m+j, r] = 1, V[j*p+1, r] = 1, W[1*n+0, r] = 1
    #            ... this doesn't work because V and W are coupled.
    #
    # Let me try a different approach. Use rank-24 with the following structure:
    # For each k in 0..5, use 4 terms (one per j):
    #   Term (j,k): U[i*m+j, r] = delta_{i,0} + delta_{i,1} = 1 for both i
    #                V[j*p+k, r] = 1
    #                W[k*n+i, r] = 1 for both i
    #
    # But this would give C[0,k] = sum_j (1) * B[j,k] * (1) = sum_j B[j,k]
    # and C[1,k] = sum_j (1) * B[j,k] * (1) = sum_j B[j,k]
    # Which is wrong - they should depend on A differently.
    #
    # Correct approach: Use 24 terms with the following pattern:
    # For each k in 0..5, use 4 terms indexed by j in 0..3:
    #   Term (j,k): 
    #     U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #     V[j*p+k, r] = 1
    #     W[k*n+0, r] = 1, W[k*n+1, r] = 1
    #
    # Check: sum_r U[i*m+j', r] V[j'*p+k', r] W[k'*n+i', r] A[i,j] B[j,k]
    # For term (j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Contribution: A[i,j] * B[j,k] * 1
    # Sum over r: sum_{j',k'} A[i,j'] * B[j',k'] * [j'=j and k'=k] 
    #             = A[i,j] * B[j,k]
    # Sum over j,k: sum_j sum_k A[i,j] B[j,k] ... but this sums over all k, not just the target k.
    #
    # I see the issue. The tensor decomposition computes the trilinear form:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # where T[i,j,k] = 1 for all i,j,k.
    #
    # So we need: sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) * (sum_{k,i} W[k*n+i,r] C[k,i])
    #
    # Let's verify the trivial rank-48 decomposition:
    # For each (i,j,k), term r = i*m*p + j*p + k:
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1
    #
    # LHS: sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # RHS: sum_{i,j,k} (A[i,j]) (B[j,k]) (C[k,i]) = same. Correct.
    #
    # Now, can we reduce rank? The tensor T[i,j,k] = 1 has a special structure.
    # Note that T[i,j,k] = T[i,j,k] for all i,j,k, so it's a rank-1 tensor in the sense that
    # T = u ⊗ v ⊗ w where u[i]=1, v[j]=1, w[k]=1? No, that's not right.
    #
    # Actually, T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    #
    # One decomposition: For each j, use 2 terms:
    #   Term (j,0): U[i*m+j, r] = 1 for i=0, V[j*p+k, r] = 1 for k=0, W[0*n+i, r] = 1 for i=0
    #   This is getting complicated.
    #
    # Let me just use the rank-48 trivial decomposition first to ensure correctness,
    # then try to reduce.
    #
    # Actually, let's use a smarter approach. The tensor T[i,j,k] = 1 has the property that
    # it doesn't depend on i, j, or k individually. We can exploit this.
    #
    # Decomposition 1: For each j, use 2 terms:
    #   Term (j, a) for a in {0, 1}:
    #     U[i*m+j, r] = 1 for all i
    #     V[j*p+k, r] = 1 for all k
    #     W[k*n+i, r] = delta_{i, a}
    #
    # Check: sum_r U[i*m+j', r] V[j'*p+k', r] W[k'*n+i', r] A[i,j] B[j,k]
    # For term (j, a): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = delta_{i,a}
    # Contribution: A[i,j] * B[j,k] * delta_{i,a}
    # Sum over r: sum_{j',a} A[i,j'] * B[j',k] * delta_{i,a}
    #             = A[i,j] * B[j,k] * sum_a delta_{i,a}
    #             = A[i,j] * B[j,k] * 2 (since a=0,1 and delta_{i,a} is 1 for one a)
    #
    # This gives 2 * A[i,j] * B[j,k], which is wrong by a factor of 2.
    #
    # Fix: Use U[i*m+j, r] = 1/2 for all i.
    # Then contribution: (1/2) * A[i,j] * B[j,k] * delta_{i,a}
    # Sum over r: (1/2) * A[i,j] * B[j,k] * sum_a delta_{i,a} = (1/2) * A[i,j] * B[j,k] * 2 = A[i,j] * B[j,k]
    #
    # But the problem requires integer/half-integer entries, so 1/2 is allowed.
    #
    # Total rank: m * 2 = 4 * 2 = 8? No, we have 2 terms per j, so rank = 2*m = 8.
    #
    # Wait, let me re-examine. For each j, we have 2 terms (a=0 and a=1).
    # Term (j, a):
    #   U[i*m+j, r] = 1/2 for i=0,1
    #   V[j*p+k, r] = 1 for k=0..5
    #   W[k*n+i, r] = 1 if i==a, else 0
    #
    # Check the trilinear form:
    # sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # = sum_{j,a} (sum_{i} (1/2) * A[i,j]) * (sum_k B[j,k]) * (sum_{k,i} C[k,i] * delta_{i,a})
    # = sum_{j,a} (1/2 * sum_i A[i,j]) * (sum_k B[j,k]) * (sum_k C[k,a])
    #
    # LHS = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    #     = sum_j (sum_i A[i,j]) * (sum_k B[j,k] * ... ) 
    #     This doesn't match. The issue is that the sums are coupled.
    #
    # Let me compute LHS more carefully:
    # LHS = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    #     = sum_j sum_i sum_k A[i,j] B[j,k] C[k,i]
    #     = sum_j (sum_i A[i,j]) * (sum_k B[j,k] C[k, ?])
    #     No, C[k,i] depends on both k and i.
    #
    # RHS = sum_{j,a} (1/2 * sum_i A[i,j]) * (sum_k B[j,k]) * (sum_k C[k,a])
    #     = sum_j (1/2 * (A[0,j] + A[1,j])) * (sum_k B[j,k]) * ((sum_k C[k,0]) + (sum_k C[k,1]))
    #     = sum_j (1/2 * (A[0,j] + A[1,j])) * (sum_k B[j,k]) * (sum_k C[k,0] + sum_k C[k,1])
    #
    # These are not equal in general. So this decomposition is incorrect.
    #
    # Let me go back to basics. The correct trivial decomposition is rank 48.
    # I'll implement that first, then try to find a lower rank.
    #
    # Actually, let me try a different approach. Use rank 24 with the following:
    # For each (j, k), use 2 terms:
    #   Term (j, k, 0): U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
    #   Term (j, k, 1): U[1*m+j, r+1] = 1, V[j*p+k, r+1] = 1, W[k*n+1, r+1] = 1
    #
    # This gives rank = 2*m*p = 48. Same as before.
    #
    # To get rank 24, I need to find a way to combine terms.
    #
    # Key insight: For fixed (j,k), the two terms are:
    #   (U[0*m+j], V[j*p+k], W[k*n+0]) and (U[1*m+j], V[j*p+k], W[k*n+1])
    #
    # These share the same V component. Can we combine them?
    #
    # Let's try: For each (j,k), use 1 term with:
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+0, r] = 1, W[k*n+1, r] = 1
    #
    # Check: Contribution to trilinear form:
    #   sum_{i,j',k'} A[i,j'] B[j',k'] C[k',i] * U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    #   For term (j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    #   Contribution: A[i,j] B[j,k] C[k,i]
    #   Sum over r: sum_{j',k'} A[i,j'] B[j',k'] C[k',i]
    #   This sums over all j' and k', not just the target j and k. Wrong.
    #
    # I think the issue is that I'm confusing the tensor decomposition with the bilinear algorithm.
    #
    # Let me look at this differently. The matrix multiplication C = A*B can be computed as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A bilinear algorithm uses rank r and computes:
    # C[i,k] = sum_{t=1}^r (sum_j A[i,j] U[j,i,t]) * (sum_j B[j,k] V[j,k,t]) * W[i,k,t]
    #
    # Wait, the standard form is:
    # C[i,k] = sum_t (sum_j A[i,j] U[t,i,j]) * (sum_j B[j,k] V[t,j,k]) * W[t,i,k]
    #
    # But the task specifies U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    #
    # The trilinear form is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = sum_t (sum_{i,j} U[i*m+j,t] A[i,j]) * (sum_{j,k} V[j*p+k,t] B[j,k]) * (sum_{k,i} W[k*n+i,t] C[k,i])
    #
    # And we need T[i,j,k] = 1 for all i,j,k.
    #
    # Let's verify the rank-48 trivial decomposition:
    # For each (i,j,k), t = i*m*p + j*p + k:
    #   U[i*m+j, t] = 1
    #   V[j*p+k, t] = 1
    #   W[k*n+i, t] = 1
    #
    # RHS = sum_{i,j,k} (A[i,j]) * (B[j,k]) * (C[k,i]) = LHS. Correct.
    #
    # Now, to reduce rank, we need to find linear dependencies among the 48 terms.
    #
    # Note that T[i,j,k] = 1 is constant. This means the tensor has a lot of symmetry.
    #
    # Let's try to find a rank-24 decomposition. One approach:
    # For each (j,k), use 2 terms that cover both i=0 and i=1:
    #   Term (j,k,0): U[0*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+0, t] = 1
    #   Term (j,k,1): U[1*m+j, t+1] = 1, V[j*p+k, t+1] = 1, W[k*n+1, t+1] = 1
    #
    # This is still rank 48. To get rank 24, we need to combine these.
    #
    # Idea: Use the fact that W[k*n+0, t] and W[k*n+1, t+1] are in different positions.
    # Can we use a single term with W having non-zero entries in both positions?
    #
    # Try: For each (j,k), use 1 term:
    #   U[0*m+j, t] = 1, U[1*m+j, t] = 1
    #   V[j*p+k, t] = 1
    #   W[k*n+0, t] = 1, W[k*n+1, t] = 1
    #
    # Check: sum_t U[i*m+j', t] V[j'*p+k', t] W[k'*n+i', t] A[i,j] B[j,k]
    # For term (j,k): U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # Contribution: A[i,j] B[j,k] * 1
    # Sum over t: sum_{j',k'} A[i,j'] B[j',k'] 
    # This is wrong - it sums over all j' and k'.
    #
    # The problem is that the tensor decomposition sums over all i,j,k, and the U,V,W
    # components are multiplied together. For the decomposition to be correct, we need:
    # sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all i,j,k.
    #
    # This is a system of n*m*p = 48 equations in r unknowns (the rank).
    # Each term t contributes a rank-1 tensor U_t ⊗ V_t ⊗ W_t.
    #
    # For the trivial decomposition, each term has a single non-zero entry in each of U, V, W.
    #
    # To reduce rank, we need to find a set of rank-1 tensors whose sum equals the all-ones tensor.
    #
    # One approach: Use the fact that the all-ones tensor can be decomposed as:
    # T[i,j,k] = sum_{a=0}^{1} sum_{b=0}^{3} sum_{c=0}^{5} delta_{i,a} delta_{j,b} delta_{k,c}
    #
    # This is the trivial rank-48 decomposition.
    #
    # Can we do better? The all-ones tensor has a special structure. Note that:
    # T[i,j,k] = 1 = (1) * (1) * (1)
    #
    # This suggests a rank-1 decomposition, but that's not possible because the indices are coupled.
    #
    # Actually, T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{a=0}^{1} delta_{i,a} * 1 * 1
    #
    # But this doesn't help because the other factors don't depend on j or k.
    #
    # Let me try a different decomposition. Use rank 24 with the following structure:
    # For each k in 0..5, use 4 terms (one per j):
    #   Term (j, k): 
    #     U[i*m+j, t] = 1 for i=0,1
    #     V[j*p+k, t] = 1
    #     W[k*n+i, t] = 1 for i=0,1
    #
    # Check: sum_t U[i*m+j', t] V[j'*p+k', t] W[k'*n+i', t]
    # For term (j,k): U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # Contribution: 1 * 1 * 1 = 1 if j'=j and k'=k, else 0.
    # Sum over t: 1 if (j'=j and k'=k) for some t, but t is indexed by (j,k), so for fixed (j',k'),
    # only term (j',k') contributes, giving 1.
    #
    # Wait, this seems correct! Let me verify more carefully.
    #
    # For fixed (i, j, k), sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # Term t corresponds to (j', k'). U[i*m+j, t] is non-zero only if j=j'.
    # V[j*p+k, t] is non-zero only if k=k'.
    # W[k*n+i, t] is non-zero only if k=k'.
    # So only term (j,k) contributes, and it gives 1*1*1 = 1.
    #
    # This works! And the rank is m*p = 4*6 = 24.
    #
    # Let me implement this.

    # For each (j, k) pair, create a term:
    #   U[i*m+j, t] = 1 for i=0,1
    #   V[j*p+k, t] = 1
    #   W[k*n+i, t] = 1 for i=0,1

    t = 0
    for j in range(m):
        for k in range(p):
            # U: set U[i*m+j, t] = 1 for i=0,1
            for i in range(n):
                U[i * m + j, t] = 1.0
            # V: set V[j*p+k, t] = 1
            V[j * p + k, t] = 1.0
            # W: set W[k*n+i, t] = 1 for i=0,1
            for i in range(n):
                W[k * n + i, t] = 1.0
            t += 1

    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = np.random.rand(p, n)

    # LHS: sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    LHS = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                LHS += A_test[i, j] * B_test[j, k] * C_test[k, i]

    # RHS: sum_t (sum_{i,j} U[i*m+j,t] A[i,j]) * (sum_{j,k} V[j*p+k,t] B[j,k]) * (sum_{k,i} W[k*n+i,t] C[k,i])
    RHS = 0.0
    for t in range(rank):
        a_dot = 0.0
        for i in range(n):
            for j in range(m):
                a_dot += U[i * m + j, t] * A_test[i, j]
        b_dot = 0.0
        for j in range(m):
            for k in range(p):
                b_dot += V[j * p + k, t] * B_test[j, k]
        c_dot = 0.0
        for k in range(p):
            for i in range(n):
                c_dot += W[k * n + i, t] * C_test[k, i]
        RHS += a_dot * b_dot * c_dot

    if not np.isclose(LHS, RHS, atol=1e-9):
        # Fallback to rank-48 trivial decomposition if verification fails
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, t] = 1.0
                    V[j * p + k, t] = 1.0
                    W[k * n + i, t] = 1.0
                    t += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
