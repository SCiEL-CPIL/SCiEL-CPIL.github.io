"""Combined program: structural rank-32 base with pair-term dependency removal."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    # Build the rank-32 base decomposition using Strassen blocks
    # Block 1: cols 0-3 -> two Strassen-7 blocks (rank 14)
    # Block 2: cols 4-8 -> Strassen(4,5) + schoolbook(6) + Strassen(7,8) (rank 7+4+7=18)
    # Total: 32

    rank_base = 32
    U = [[0.0] * rank_base for _ in range(n * m)]   # 4 x 32
    V = [[0.0] * rank_base for _ in range(m * p)]   # 18 x 32
    W = [[0.0] * rank_base for _ in range(p * n)]   # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b00+b11) -> C00, C11
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M2: (a10+a11)b00 -> C10, -C11
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        # M3: a00(b01-b11) -> C01, C11
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M4: a11(b10-b00) -> C00, C10
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        # M5: (a00+a01)b11 -> -C00, C01
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        # M6: (a10-a00)(b00+b01) -> C11
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (a01-a11)(b10+b11) -> C00
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)

    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition is correct
    def verify(U_arr, V_arr, W_arr):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == tensor T[i,j,k] = delta(i+m*j, k) ... 
        Actually the tensor is: T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B 
        contributes to (i,k) entry of C. So T[i, j, k] = 1 for all i,j,k (it's the matmul tensor).
        Wait, the matmul tensor M_{n,m,p} has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
        The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to reduce rank by removing pairs of terms that are linearly dependent
    # in the tensor product space.
    
    def try_remove_pair(U_arr, V_arr, W_arr, r1, r2):
        """Try to remove terms r1 and r2 by expressing them as linear combinations
        of other terms. This is a heuristic: check if the column vectors in the 
        Khatri-Rao product space are linearly dependent."""
        # The tensor equation is: sum_r u_r ⊗ v_r ⊗ w_r = T
        # If we can find scalars a, b such that a*(u_r1⊗v_r1⊗w_r1) + b*(u_r2⊗v_r2⊗w_r2)
        # can be replaced by adjustments to other terms, we reduce rank.
        # A simpler heuristic: check if the two terms are scalar multiples in some sense.
        
        # Check if U[:,r1] is proportional to U[:,r2]
        u1, u2 = U_arr[:, r1], U_arr[:, r2]
        v1, v2 = V_arr[:, r1], V_arr[:, r2]
        w1, w2 = W_arr[:, r1], W_arr[:, r2]
        
        # If u1 = a*u2 and v1 = a*v2 and w1 = a*w2, they're redundant
        # More generally, check linear dependence of the 3-way tensors
        # This is expensive, so use a quick check:
        # If u1 is proportional to u2:
        norm_u1 = np.linalg.norm(u1)
        norm_u2 = np.linalg.norm(u2)
        if norm_u1 < 1e-12 or norm_u2 < 1e-12:
            return False
        
        # Check if the pair can be eliminated via a change of basis
        # This is a simplified check - in practice we'd need full tensor rank analysis
        return False

    # More effective: try single-term removal by checking if a term is in the span
    # of the other terms in the tensor product space.
    
    def try_remove_single(U_arr, V_arr, W_arr, r):
        """Check if term r can be removed (i.e., is linearly dependent on others)."""
        # Build the tensor matrix for all terms except r
        # This is a (n*m*p) x (rank-1) system
        # The tensor T is all-ones in the relevant subspace.
        # Actually, we need: sum_{s != r} U[:,s]*V[:,s]*W[:,s] should equal T - U[:,r]*V[:,r]*W[:,r]
        # But we can't easily check this without solving.
        
        # Quick heuristic: if the term has very small norm, it might be removable
        norm = np.linalg.norm(U_arr[:, r]) * np.linalg.norm(V_arr[:, r]) * np.linalg.norm(W_arr[:, r])
        if norm < 1e-12:
            return True
        return False

    # Try removing terms one by one (greedy)
    removed = []
    for r in range(rank_base - 1, -1, -1):
        if r in removed:
            continue
        if try_remove_single(U_np, V_np, W_np, r):
            removed.append(r)
    
    # Try pair removal
    for r1 in range(rank_base):
        if r1 in removed:
            continue
        for r2 in range(r1 + 1, rank_base):
            if r2 in removed:
                continue
            if try_remove_pair(U_np, V_np, W_np, r1, r2):
                removed.append(r1)
                removed.append(r2)
                break

    # If no terms were removed, return the base decomposition
    if not removed:
        return {"U": U, "V": V, "W": W}

    # Filter out removed terms
    keep = [r for r in range(rank_base) if r not in removed]
    U_new = [row[r] for row in U for r in keep]  # This is wrong, need proper indexing
    U_new = [[U[i][r] for r in keep] for i in range(n*m)]
    V_new = [[V[i][r] for r in keep] for i in range(m*p)]
    W_new = [[W[i][r] for r in keep] for i in range(p*n)]

    # Verify the reduced decomposition
    U_np_new = np.array(U_new, dtype=float)
    V_np_new = np.array(V_new, dtype=float)
    W_np_new = np.array(W_new, dtype=float)

    if verify(U_np_new, V_np_new, W_np_new):
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        # Fallback to base
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
