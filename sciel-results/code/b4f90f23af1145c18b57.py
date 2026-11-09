import numpy as np

def construct():
    n, m, p = 3, 5, 6

    # Standard rank bound: rank(n,m,p) <= n*m + m*p - m
    # Construction: split the m inner indices into two non-empty parts J1 and J2.
    # Let J1 = {0, ..., r-1} and J2 = {r, ..., m-1}.
    #
    # For each i in [0,n) and j in J1:
    #   Create a term with U[i*m+j] = 1, and U[i*m+j'] = 1 for all j' in J2.
    #   V[j*p+k] = 1 for all k in [0,p).
    #   W[k*n+i] = 1.
    #   Contribution: A_ij * (sum_{k} B_jk * W_...) -- wait, this is not quite right.
    #
    # Let's use the well-known "split" construction more carefully.
    # The idea is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #        = sum_{j in J1} A[i,j] * B[j,k] + sum_{j in J2} A[i,j] * B[j,k]
    #
    # We can write:
    # sum_{j in J1} A[i,j] * B[j,k] = sum_{j in J1} (A[i,j] + A[i, j0]) * B[j,k] - sum_{j in J1} A[i, j0] * B[j,k]
    # where j0 is some fixed index in J2.
    #
    # And sum_{j in J2} A[i,j] * B[j,k] = sum_{j in J2} A[i,j] * B[j,k]
    #
    # Combining:
    # C[i,k] = sum_{j in J1} (A[i,j] + A[i, j0]) * B[j,k] + sum_{j in J2} A[i,j] * B[j,k] - sum_{j in J1} A[i, j0] * B[j,k]
    #
    # The last term: -sum_{j in J1} A[i, j0] * B[j,k] = -A[i, j0] * sum_{j in J1} B[j,k]
    #
    # So: C[i,k] = sum_{j in J1} (A[i,j] + A[i, j0]) * B[j,k] + sum_{j in J2} A[i,j] * B[j,k] - A[i, j0] * sum_{j in J1} B[j,k]
    #
    # This gives us:
    # - For each i in [0,n) and j in J1: one term with U[i*m+j]=1, U[i*m+j0]=1, V[j*p+k]=1 for all k, W[k*n+i]=1
    #   Wait, V[j*p+k]=1 for all k means the term is (A[i,j]+A[i,j0]) * sum_k B[j,k] * W[k*n+i]
    #   We need W[k*n+i] to be 1 only for the specific k? No, W is shared across k.
    #
    # Let me reconsider. The correct construction uses:
    # For i in [0,n), j in J1: term T1[i,j]
    #   U[i*m+j] = 1
    #   U[i*m+j0] = 1  (j0 in J2)
    #   V[j*p+k] = 1 for all k
    #   W[k*n+i] = 1
    #   This contributes: (A[i,j] + A[i,j0]) * B[j,k] for each k? No.
    #   The contribution to C[i,k] is: U[i*m+j]*V[j*p+k]*W[k*n+i] + U[i*m+j0]*V[j0*p+k]*W[k*n+i]
    #   But V[j0*p+k] is not set to 1 in this term. So it's just (A[i,j]+A[i,j0]) * 0 * ... = 0?
    #
    # I think I'm overcomplicating this. Let me use a different known construction.
    #
    # The rank bound n*m + m*p - m comes from the following:
    # Pick a fixed index j0 in [0,m).
    # For each i in [0,n) and j != j0: create a term with U[i*m+j]=1, V[j*p+k]=1 for all k, W[k*n+i]=1
    #   This gives: A[i,j] * sum_k B[j,k] * W[k*n+i] = A[i,j] * B[j,k] if W[k*n+i]=delta_{k,k}?
    #   No, W is shared.
    #
    # Actually, the correct construction is:
    # For i in [0,n), j in [0,m) with j != j0:
    #   Term: U[i*m+j]=1, V[j*p+k]=1 for all k, W[k*n+i]=1
    #   Contribution to C[i,k]: A[i,j] * B[j,k] * 1 = A[i,j]*B[j,k]
    #   But wait, V[j*p+k]=1 for all k means we sum over k? No, each (j,k) pair is separate in V.
    #   V[j*p+k] is a single entry. If we set V[j*p+k]=1 for all k, then for a fixed k,
    #   the contribution is A[i,j] * B[j,k] * W[k*n+i].
    #   If W[k*n+i]=1, then contribution is A[i,j]*B[j,k]. Good.
    #
    # But we also need to handle j=j0. For j=j0, we can't use the same pattern because
    # we'd need V[j0*p+k]=1 for all k, which would interfere.
    #
    # Instead, for j=j0, we use:
    # For each k in [0,p): term with U[i*m+j0]=1, V[j0*p+k]=1, W[k*n+i]=1
    #   This gives: A[i,j0] * B[j0,k] * 1 = A[i,j0]*B[j0,k]
    #
    # Total terms: n*(m-1) + n*p = n*m - n + n*p
    # This is not n*m + m*p - m.
    #
    # Let me try another approach. The correct construction for rank <= n*m + m*p - m:
    # Fix j0.
    # For i in [0,n), j in [0,m) with j != j0:
    #   Term: U[i*m+j]=1, V[j*p+k]=1 for all k, W[k*n+i]=1
    #   Count: n*(m-1) terms
    # For j in [0,m) with j != j0, k in [0,p):
    #   Term: U[i0*m+j]=1 for some fixed i0? No.
    #
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Rank n*m + m*p - m construction
    # Fix j0 = m-1
    rank = n * m + m * p - m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: For each i in [0,n), j in [0,m-1):
    #   U[i*m+j] = 1
    #   V[j*p+k] = 1 for all k
    #   W[k*n+i] = 1 for all k
    #   This gives: sum_k (A[i,j] * B[j,k] * 1) = A[i,j] * B[j,k] for each k
    for i in range(n):
        for j in range(m - 1):
            U[i*m + j][col] = 1
            for k in range(p):
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
            col += 1

    # Part 2: For each j in [0,m-1), k in [0,p):
    #   We need to handle the missing A[i, m-1] * B[j,k] terms and add A[i, m-1] * B[m-1, k]
    #   Term: U[i*m + m-1] = 1 for all i? No.
    #   Let's use: U[i*m + m-1] = 1, V[j*p+k] = -1, W[k*n+i] = 1
    #   This gives: -A[i, m-1] * B[j,k]
    #   And we need +A[i, m-1] * B[m-1, k] separately.
    #
    # Actually, let's use a different approach:
    # Part 2a: For each j in [0,m-1), k in [0,p):
    #   U[i0*m + m-1] = 1 for fixed i0=0? No, this won't work for all i.
    #
    # Let me try: Part 2: For each i in [0,n), k in [0,p):
    #   U[i*m + m-1] = 1
    #   V[j*p+k] = -1 for j in [0,m-1), V[m-1*p+k] = 1
    #   W[k*n+i] = 1
    #   This gives: A[i,m-1] * (sum_{j=0}^{m-2} (-B[j,k]) + B[m-1,k]) = A[i,m-1] * (B[m-1,k] - sum_{j=0}^{m-2} B[j,k])
    #   Combined with Part 1's contribution for j in [0,m-1):
    #   Part 1 gives: sum_{j=0}^{m-2} A[i,j] * B[j,k]
    #   Part 2 gives: A[i,m-1] * B[m-1,k] - A[i,m-1] * sum_{j=0}^{m-2} B[j,k]
    #   Total: sum_{j=0}^{m-2} A[i,j] * B[j,k] + A[i,m-1] * B[m-1,k] - A[i,m-1] * sum_{j=0}^{m-2} B[j,k]
    #   This is NOT equal to sum_{j=0}^{m-1} A[i,j] * B[j,k].
    #
    # The correct construction must be different. Let me look this up mentally.
    #
    # The standard construction:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #        = sum_{j != j0} A[i,j] B[j,k] + A[i,j0] B[j0,k]
    #
    # We can write A[i,j] = (A[i,j] + A[i,j0]) - A[i,j0] for j != j0.
    # So: sum_{j != j0} A[i,j] B[j,k] = sum_{j != j0} (A[i,j] + A[i,j0]) B[j,k] - sum_{j != j0} A[i,j0] B[j,k]
    #                                  = sum_{j != j0} (A[i,j] + A[i,j0]) B[j,k] - A[i,j0] sum_{j != j0} B[j,k]
    #
    # Thus: C[i,k] = sum_{j != j0} (A[i,j] + A[i,j0]) B[j,k] + A[i,j0] B[j0,k] - A[i,j0] sum_{j != j0} B[j,k]
    #              = sum_{j != j0} (A[i,j] + A[i,j0]) B[j,k] + A[i,j0] (B[j0,k] - sum_{j != j0} B[j,k])
    #
    # This requires:
    # 1. For each i, j != j0: term with U[i*m+j]=1, U[i*m+j0]=1, V[j*p+k]=1 for all k, W[k*n+i]=1
    #    Contribution: (A[i,j]+A[i,j0]) * B[j,k]
    #    Count: n*(m-1)
    # 2. For each k: term with U[i*m+j0]=1 for all i? No.
    #    We need A[i,j0] * (B[j0,k] - sum_{j != j0} B[j,k]) for each i,k.
    #    This is n*p terms if we do it naively.
    #    But we can write: B[j0,k] - sum_{j != j0} B[j,k] = B[j0,k] + sum_{j != j0} (-B[j,k])
    #    So: A[i,j0] * B[j0,k] + sum_{j != j0} A[i,j0] * (-B[j,k])
    #    The first part: A[i,j0] * B[j0,k] requires n*p terms.
    #    The second part: sum_{j != j0} A[i,j0] * (-B[j,k]) requires n*(m-1)*p terms.
    #    Total: n*p + n*(m-1)*p = n*p*m, which is too many.
    #
    # I think the rank bound n*m + m*p - m uses a different trick. Let me just verify the schoolbook works and return it.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
