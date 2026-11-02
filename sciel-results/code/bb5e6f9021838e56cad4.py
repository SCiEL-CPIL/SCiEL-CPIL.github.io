import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 7

    # Build the 50-rank decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the remaining 2x2x1 block
    U = [[0] * 50 for _ in range(n * m)]  # 8 x 50
    V = [[0] * 50 for _ in range(m * p)]  # 28 x 50
    W = [[0] * 50 for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= 50:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= 50:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
        
        return True

    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            if not add_strassen_222(0, j_start, k_start):
                break
        if time.time() > deadline:
            break
        if not add_schoolbook_221(0, j_start, 6):
            break

    # Now try to reduce rank via tensor mode unfolding and null-space projection
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Strategy: try to find a lower-rank decomposition by exploiting
    # the structure of the multiplication tensor.
    # 
    # The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # 
    # Key observation: for fixed j, the sub-tensor T[:, j*p+*, *] has a
    # specific structure. Let's try to find linear dependencies.
    #
    # Approach: For each j, the "slice" of the tensor where the middle
    # index has j*p+k form gives us a matrix. If we can find that some
    # combination of Strassen terms cancels out, we can reduce rank.
    #
    # More concretely: try to find a rank-r decomposition with r < 50
    # by solving the tensor equation directly using the structure.
    
    # Let's try a direct approach: assume rank r and solve for the
    # decomposition using the known structure of the tensor.
    #
    # The tensor has 56 non-zero entries. For a rank-r decomposition:
    # sum_{s=1}^{r} U[a,s] * V[b,s] * W[c,s] = T[a,b,c]
    #
    # This is a system of 56 equations (for the non-zero entries) plus
    # constraints that all other entries are zero.
    #
    # A simpler approach: try to find if any of the 50 terms can be
    # eliminated by a change of basis in the rank space.
    
    # Try: can we find a 49-rank decomposition?
    # Check if the 50 columns of U, V, W have a tensor linear dependency.
    # This is hard to check directly, so let's try a different approach.
    
    # Alternative: use the fact that the multiplication tensor for
    # M_{2,4,7} can potentially be decomposed with rank 49 or less
    # by using a more clever block decomposition.
    
    # Let's try: instead of 2 Strassen blocks for each j, use a
    # different partitioning of the 4x7 B matrix.
    
    # Actually, let's try to verify if rank 50 is truly minimal by
    # checking the border rank or using a certificate.
    
    # For now, let's try to find a 49-rank decomposition by
    # attempting to merge two schoolbook terms into one Strassen-like term.
    
    # The 2 schoolbook blocks contribute 8 terms. Can we replace them
    # with 7 terms using a modified Strassen?
    
    # For a 2x2x1 product (2 rows, 2 cols of A times 2x1 of B),
    # the minimum rank is 4 (schoolbook). But if we can share
    # structure with adjacent blocks, we might save.
    
    # Let's try a different approach: use a 3x3 Strassen-like
    # decomposition for the 2x3 sub-problems.
    
    # Actually, the cleanest approach: try to directly solve for a
    # rank-r decomposition with r < 50 using the tensor structure.
    
    # The tensor T can be written as:
    # T = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{6} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # where e_{i,j} is the unit vector for (i,j) in the first mode, etc.
    
    # For rank r, we need:
    # T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s
    
    # Let's try to find such a decomposition with r < 50.
    
    # Key insight: the tensor has a "cyclic" structure in the indices.
    # For each (i,j,k), the entry is at position (i*m+j, j*p+k, k*n+i).
    
    # Let's try to use the null space of the matricization to find
    # a lower-rank decomposition.
    
    # Matricize T as a (n*m) x (m*p*p*n) matrix... that's too large.
    # Instead, fix one mode and look at the other two.
    
    # Fix the first mode: T_a[b,c] for a = i*m+j
    # For a given a = i*m+j, T_a[b,c] = 1 iff b = j*p+k and c = k*n+i
    # So T_a has non-zero entries at (j*p+k, k*n+i) for k=0..6
    # That's 7 non-zero entries per row of the matricization.
    
    # The matricization T_{(1)} is an 8 x (28*14) = 8 x 392 matrix.
    # Its rank is at most 8, but the tensor rank is different.
    
    # Let's try a more direct approach: use the known structure to
    # set up a system of equations for a rank-r decomposition.
    
    # For rank r, we have r * (8 + 28 + 14) = 40r unknowns.
    # We have 56 non-zero equations (must equal 1) and many zero equations.
    
    # This is a nonlinear system. Let's try to solve it for small r
    # starting from r=49 and going down.
    
    # Actually, let's try a smarter approach: exploit the block structure.
    
    # The 50-term decomposition has the following structure:
    # - 6 Strassen blocks (42 terms) for the 2x2x2 sub-problems
    # - 2 schoolbook blocks (8 terms) for the 2x2x1 sub-problems
    
    # Can we do better by using a different block decomposition?
    
    # What if we use a 2x4x4 block with rank 21 (instead of 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1)?
    # 2x4x4: schoolbook is 32, Strassen-based would be... 
    # Actually 2x4x4 can be done with rank 2*4*4/2 = 16? No, that's not right.
    
    # Let's try: 2x4x7 = 2x(4x7). The 4x7 part can be decomposed as
    # 4x7 = (2x7) + (2x7) in the j-dimension.
    # Each 2x7 can be done with... 2x7 schoolbook is 14, Strassen-based?
    
    # Actually, let's just try to verify if 50 is minimal by checking
    # if we can find a 49-rank decomposition.
    
    # Try: remove one of the schoolbook terms and see if the remaining
    # 49 terms can be adjusted to still give the correct tensor.
    
    # This is a complex optimization problem. Let's try a simpler check:
    # is the 50-rank decomposition "tight" in the sense that no term
    # can be removed?
    
    # For each term t, check if T - u_t ⊗ v_t ⊗ w_t can be decomposed
    # with rank 49. This is equivalent to checking if the remaining
    # tensor has rank ≤ 49.
    
    # A necessary condition: the matricization of T - u_t ⊗ v_t ⊗ w_t
    # must have rank ≤ 49.
    
    # Let's check this for a few terms.
    
    def tensor_rank_upper_bound(U_mat, V_mat, W_mat):
        """Return an upper bound on the tensor rank via matricization."""
        # Matricize along mode 1: (n*m) x (m*p * p*n)
        # This is too large. Use a smaller matricization.
        # Matricize along mode 1: (n*m) x (m*p) * (p*n) -> use (n*m) x (m*p)
        # by summing over the third mode... no, that doesn't work.
        
        # Better: check the rank of the "slice" matrices.
        # For each value of the second index b, T[:, b, :] is a matrix.
        # The tensor rank is at least the max rank of any slice.
        
        # Actually, let's just check if the decomposition is valid.
        return None
    
    # Let's try a completely different approach: use the known result
    # that the rank of M_{n,m,p} is at least n*p (from the "slice" bound).
    # For n=2, p=7: rank >= 14. That's a very weak bound.
    
    # A better bound: rank >= max(n*m, m*p, p*n) / min(n, m, p)
    # = max(8, 28, 14) / 2 = 14. Still weak.
    
    # The best known lower bound for M_{2,4,7} is probably 50 or close to it.
    
    # Let's just return the 50-rank decomposition and verify it.
    
    # Verify the decomposition
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(col):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check some zero entries
        for a in range(n * m):
            for b in range(m * p):
                j_from_a = a % m
                i_from_a = a // m
                j_from_b = b // p
                k_from_b = b % p
                if j_from_a != j_from_b:
                    continue
                for c in range(p * n):
                    i_from_c = c % n
                    k_from_c = c // n
                    if i_from_a != i_from_c or k_from_b != k_from_c:
                        s = 0.0
                        for r in range(col):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if abs(s) > 1e-9:
                            return False
        return True
    
    ok = verify()
    if not ok:
        # Fallback: return schoolbook decomposition (rank 56)
        U = [[0] * 56 for _ in range(n * m)]
        V = [[0] * 56 for _ in range(m * p)]
        W = [[0] * 56 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
