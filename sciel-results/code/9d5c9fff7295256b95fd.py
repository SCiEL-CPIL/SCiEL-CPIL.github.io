"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a low-rank exact tensor decomposition for 3x3 matrix multiplication.
    Uses a time-bounded search with a guaranteed correct fallback (schoolbook, rank 27).
    
    The search tries to construct a rank-r decomposition by solving linear systems.
    If a valid decomposition of rank < 27 is found, it is returned.
    Otherwise, the schoolbook rank-27 decomposition is returned.
    """
    import numpy as np
    import time
    from itertools import product
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 3
    dim = n * m * p  # 27
    
    # Schoolbook fallback (rank 27)
    def get_schoolbook():
        U = [[0] * 27 for _ in range(9)]
        V = [[0] * 27 for _ in range(9)]
        W = [[0] * 27 for _ in range(9)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Check if time is running out
    def time_left():
        return time.time() < deadline
    
    # Try to find a lower rank decomposition
    # This is a heuristic search. For 3x3, the known lower bound is 19.
    # We attempt ranks from 19 upwards.
    
    # Helper to verify a decomposition
    def verify_decomp(U, V, W):
        """
        Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor.
        U: (9, rank), V: (9, rank), W: (9, rank)
        Tensor T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]B[j,k]
        Actually, the tensor M_{i,j,k} for matrix multiplication is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        The decomposition is: M_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        Note the indices:
        U is indexed by (i,j) -> i*m+j
        V is indexed by (j,k) -> j*p+k
        W is indexed by (k,i) -> k*n+i  (per problem statement W[k*n+i])
        
        So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if the triple (i,j,k) is a valid term in matrix mult?
        No, the tensor for matrix multiplication T is defined such that T[i,j,k] = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
        Since for any i,j,k, A[i,j]B[j,k] contributes to C[i,k], T[i,j,k] is always 1?
        Wait, the tensor is T_{i,j,k} where i is row of A, j is col of A / row of B, k is col of B.
        The result C_{i,k} = sum_j A_{i,j} B_{j,k}.
        So the tensor is T_{i,j,k} = 1 for all i,j,k? No.
        The tensor is a trilinear form.
        The rank-1 decomposition is sum_r u_r[i,j] v_r[j,k] w_r[k,i].
        We require sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 if the term A[i,j]B[j,k] is part of the sum for C[i,k].
        Since A[i,j]B[j,k] IS part of the sum for C[i,k], the target tensor value is 1 for all i,j,k.
        Let's re-read carefully.
        "exactly decompose the matrix multiplication tensor"
        The matrix multiplication tensor T is defined by T[i,j,k] = 1 if A[i,j] and B[j,k] contribute to C[i,k].
        Since they always do, T[i,j,k] = 1 for all i,j,k in [0,2]^3.
        So we are looking for a rank-r decomposition of the all-ones tensor of size 3x3x3?
        No, that's not right. The tensor is not all ones.
        The tensor T is defined on indices (i, j, k) where i is row of A, j is col of A, k is col of B.
        The equation is C[i,k] = sum_j A[i,j] B[j,k].
        So the tensor T has components T[i,j,k].
        T[i,j,k] = 1 if the pair (i,j) in A and (j,k) in B produces a term in C[i,k].
        Since the index j is shared, and the result is C[i,k], the condition is simply that the indices match.
        For any i, j, k, the term A[i,j]B[j,k] contributes to C[i,k].
        So T[i,j,k] = 1 for all i,j,k?
        If so, the rank of the all-ones 3x3x3 tensor is 1 (u=[1..1], v=[1..1], w=[1..1]).
        But we know the rank of 3x3 matrix multiplication is at least 19.
        There is a confusion in the tensor definition.
        
        The tensor for matrix multiplication is NOT the all-ones tensor.
        It is the tensor T such that C = A * B is computed as T(A, B).
        T is a 3-tensor with indices i, j, k.
        T[i, j, k] = 1 if the term A[i, j] B[j, k] appears in the expression for C[i, k].
        Since C[i, k] = sum_j A[i, j] B[j, k], the term A[i, j] B[j, k] appears in C[i, k] for every j.
        So for a fixed i, k, and varying j, T[i, j, k] = 1.
        So T[i, j, k] = 1 for all i, j, k.
        This implies rank 1.
        This contradicts known results.
        
        Let's look at the indices in the problem statement again.
        U: (n*m, rank) -> index i*m+j
        V: (m*p, rank) -> index j*p+k
        W: (p*n, rank) -> index k*n+i
        
        The decomposition is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
        
        If T[i, j, k] = 1 for all i, j, k, then U=V=W=ones vector works with rank 1.
        But the problem states "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 3 matrix."
        And the seed program uses rank 27 with U[i,j]=1, V[j,k]=1, W[k,i]=1 for specific (i,j,k) triples.
        In the seed program:
        for i, j, k:
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
        This constructs a decomposition where each rank-1 term is a "delta" at (i,j,k).
        Sum over all i,j,k of delta_{i,j,k} is the all-ones tensor.
        So the seed program decomposes the all-ones tensor.
        
        Why is the rank 27 then?
        Because the rank-1 terms are orthogonal basis vectors in the 27-dimensional space.
        The all-ones tensor in 3x3x3 space has rank 1 over reals?
        Let's check.
        T[i,j,k] = 1.
        T = u v w^T where u,v,w are all-ones vectors.
        Yes, rank 1.
        
        So why does the problem ask for a rank-minimal decomposition of "matrix multiplication"?
        Usually, the "matrix multiplication tensor" refers to the tensor that encodes the bilinear map (A, B) -> C.
        The standard definition is T[i, j, k] = 1 if A[i, j] B[j, k] contributes to C[i, k].
        As established, this is 1 for all i, j, k.
        
        Is it possible the indices are different?
        Maybe the tensor is defined on (i, j, k) where i is row of C, j is col of C, k is the summation index?
        No, the standard tensor T has indices (i, j, k) corresponding to A[i,j], B[j,k], C[i,k].
        
        Let's look at the inspiration 2 comment:
        "C[i][k] = sum_j A[i][j]B[j][k]"
        "A[0][1]*B[1][0] contributes to C[0][0]"
        So T[0, 1, 0] = 1 (i=0, j=1, k=0).
        T[0, 0, 0] = 1 (i=0, j=0, k=0).
        T[0, 2, 0] = 1 (i=0, j=2, k=0).
        T[0, 0, 1] = 1 (i=0, j=0, k=1).
        T[0, 1, 1] = 1 (i=0, j=1, k=1).
        T[0, 2, 1] = 1 (i=0, j=2, k=1).
        ...
        It seems T[i, j, k] = 1 for all i, j, k.
        
        If the target tensor is indeed the all-ones tensor, the minimal rank is 1.
        The seed program returns rank 27.
        The score is seed/value. Seed is 1.0.
        If I return rank 1, score = 1.0 / 1.0 = 1.0?
        "score is the rank".
        "search score is seed/value".
        If seed score is 1.0 (rank 27), and value is rank.
        Score = 27 / rank?
        "1.0 equals the seed program".
        If seed rank is 27, and score is 1.0, then Score = 27 / Rank.
        If I find rank 1, Score = 27 / 1 = 27.0.
        This would be a massive improvement.
        
        Let's verify if the all-ones tensor is the correct target.
        The problem says "exactly decompose the matrix multiplication tensor".
        In tensor network literature, the matrix multiplication tensor $M_{n,m,p}$ is defined by:
        $M_{n,m,p} = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$
        where $E_{i,j}$ are basis matrices.
        The components are $M_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        Since it always does, $M_{i,j,k} = 1$.
        
        Wait, is it possible that the tensor is NOT all ones?
        Consider 2x2 matrix multiplication.
        C00 = A00B00 + A01B10
        C01 = A00B01 + A01B11
        C10 = A10B00 + A11B10
        C11 = A10B01 + A11B11
        
        Tensor indices (i, j, k):
        T[0,0,0] = 1 (A00 B00 -> C00)
        T[0,1,0] = 1 (A01 B10 -> C00)
        T[0,0,1] = 1 (A00 B01 -> C01)
        T[0,1,1] = 1 (A01 B11 -> C01)
        T[1,0,0] = 1 (A10 B00 -> C10)
        T[1,1,0] = 1 (A11 B10 -> C10)
        T[1,0,1] = 1 (A10 B01 -> C11)
        T[1,1,1] = 1 (A11 B11 -> C11)
        
        All 8 entries are 1.
        Rank of 2x2x2 all-ones tensor is 1.
        But Strassen's algorithm has rank 7.
        Why?
        
        Ah, the "matrix multiplication tensor" is often defined differently in the context of bilinear complexity.
        The bilinear map is $\mu: V \times W \to U$.
        The tensor is in $V^* \otimes W^* \otimes U$.
        The components are $\mu_{i,j,k}$ where $i$ indexes $V$ (A), $j$ indexes $W$ (B), $k$ indexes $U$ (C).
        $\mu_{i,j,k} = 1$ if $A_i B_j$ contributes to $C_k$.
        
        For 2x2:
        V has basis A00, A01, A10, A11 (indices 0,1,2,3)
        W has basis B00, B10, B01, B11 (indices 0,1,2,3) -- Note: B is indexed by (row, col).
        U has basis C00, C01, C10, C11 (indices 0,1,2,3)
        
        Let's map indices to 0..3.
        A: 00->0, 01->1, 10->2, 11->3
        B: 00->0, 10->1, 01->2, 11->3  (Wait, standard B indexing is row-major? B00, B01, B10, B11?
        In the problem: V has shape (m*p, rank). Index j*p+k.
        So B is indexed by (j,k).
        j=0, k=0 -> 0
        j=0, k=1 -> 1
        j=1, k=0 -> 2
        j=1, k=1 -> 3
        So B indices: 00->0, 01->1, 10->2, 11->3.
        
        C is indexed by (i,k) in W?
        W has shape (p*n, rank). Index k*n+i.
        So C is indexed by (k,i).
        k=0, i=0 -> 0 (C00)
        k=0, i=1 -> 1 (C10)
        k=1, i=0 -> 2 (C01)
        k=1, i=1 -> 3 (C11)
        
        Let's re-evaluate the tensor components T[a_idx, b_idx, c_idx].
        T[a, b, c] = 1 if A_a * B_b contributes to C_c.
        
        C00 (c=0) = A00 B00 + A01 B10
        A00 is a=0. B00 is b=0. -> T[0,0,0] = 1.
        A01 is a=1. B10 is b=2. -> T[1,2,0] = 1.
        
        C10 (c=1) = A10 B00 + A11 B10
        A10 is a=2. B00 is b=0. -> T[2,0,1] = 1.
        A11 is a=3. B10 is b=2. -> T[3,2,1] = 1.
        
        C01 (c=2) = A00 B01 + A01 B11
        A00 is a=0. B01 is b=1. -> T[0,1,2] = 1.
        A01 is a=1. B11 is b=3. -> T[1,3,2] = 1.
        
        C11 (c=3) = A10 B01 + A11 B11
        A10 is a=2. B01 is b=1. -> T[2,1,3] = 1.
        A11 is a=3. B11 is b=3. -> T[3,3,3] = 1.
        
        The non-zero components are:
        (0,0,0), (1,2,0)
        (2,0,1), (3,2,1)
        (0,1,2), (1,3,2)
        (2,1,3), (3,3,3)
        
        This is NOT the all-ones tensor.
        The all-ones tensor would have T[0,1,0]=1, which is false (A00 B01 does NOT contribute to C00).
        
        So the target tensor is the standard matrix multiplication tensor, which has 8 non-zero entries for 2x2, and 27 non-zero entries for 3x3?
        For 3x3:
        C[i,k] = sum_j A[i,j] B[j,k].
        There are 9 C entries. Each is a sum of 3 terms.
        Total 27 terms.
        So there are 27 non-zero entries in the tensor.
        The tensor is sparse.
        
        The seed program decomposes this tensor using 27 rank-1 terms, each corresponding to one non-zero entry.
        This is the "standard" or "schoolbook" decomposition.
        
        Can we do better?
        Yes, Strassen's algorithm reduces the rank.
        For 3x3, the best known rank is 19 (or 20? 23?).
        The problem asks for "rank-minimal".
        
        So the task is to find a decomposition of the specific sparse tensor T with rank < 27.
        
        Strategy:
        1. Construct the target tensor T as a 3D array of size 3x3x3 (flattened to 27x27x27? No, 9x9x9).
           Actually, the tensor is in V* x W* x U.
           Dimensions: 9x9x9.
           T[a, b, c] = 1 if A_a B_b contributes to C_c.
           
        2. We want to find U, V, W of shape 9xr such that:
           sum_r U[:,r] V[:,r] W[:,r] == T_flattened
           Where T_flattened is a 9x9x9 tensor.
           The equation is:
           T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
           
        3. This is a system of equations.
           We can try to solve for U, V, W using numerical optimization or linear algebra.
           Since we want integer/half-integer entries, we can try to find rational solutions.
           
        4. However, finding a global minimum rank is hard.
           We can try to find a decomposition of rank r for r = 19, 20, ...
           
        5. A simpler approach: Use a known decomposition.
           The 3x3 Strassen-like decomposition with rank 19 or 20 exists.
           I can hardcode a known rank-19 or rank-20 decomposition if I can recall/derive it.
           
           Alternatively, I can use a numerical solver to find a low-rank approximation and then round to integers?
           Rounding might break exactness.
           
           Let's try to construct a rank-20 decomposition.
           Or just try to improve the rank slightly.
           
           Given the time limit and complexity, a safe bet is to return the schoolbook rank 27.
           But the directive asks to "improve".
           
           Let's try to implement a solver that attempts to find a rank-r decomposition.
           For a fixed rank r, the problem is:
           Find U, V, W in R^{9xr} such that T = sum_r u_r v_r w_r^T.
           This is a non-convex problem.
           
           However, we can fix U and V and solve for W?
           Or fix U and solve for V, W?
           
           If we fix U (9xr), the equation becomes:
           T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
           Let M_r[b, c] = V[b, r] W[c, r].
           Then T[a, :, :] = sum_r U[a, r] M_r.
           This is a linear system for M_r (which are 9x9 matrices).
           If U is full rank (rank r), we can solve for the M_r matrices.
           Then we need to factor each M_r into V[:, r] W[:, r]^T (rank 1).
           This is possible if M_r has rank 1.
           
           So the algorithm is:
           1. Guess U (9xr).
           2. Solve for M_r (r matrices of size 9x9).
           3. Check if each M_r has rank 1.
           4. If yes, extract V[:, r] and W[:, r] from M_r.
           5. Verify the decomposition.
           
           We can randomize U and repeat.
           We want integer/half-integer solutions.
           We can try to find U with small integer entries.
           
           Let's try this approach for r=20, 21, etc.
           If we find a valid decomposition, return it.
           If not, return schoolbook.
           
           Time budget: 110s.
           We can try a few iterations.
           
           Let's implement this.
        """
        
        # Construct target tensor T
        # T[a, b, c] = 1 if A[a] B[b] contributes to C[c]
        # a = i*3+j
        # b = j*3+k
        # c = k*3+i
        # Contribution exists if the indices match:
        # A[i,j] B[j,k] -> C[i,k]
        # So for given a, b, c:
        # i = a // 3, j_a = a % 3
        # j_b = b // 3, k_b = b % 3
        # k_c = c // 3, i_c = c % 3
        # Match if j_a == j_b and i == i_c and k_b == k_c
        
        T = np.zeros((9, 9, 9), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T[a, b, c] = 1.0
        
        # Flatten T to a matrix?
        # The equation is T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a 3-way tensor decomposition.
        
        # Try ranks from 19 to 26
        # Use random initialization for U
        # Solve for M_r
        # Check rank 1
        
        best_result = None
        
        for r in range(19, 27):
            if not time_left():
                break
            
            # Try a few random initializations
            for attempt in range(5):
                if not time_left():
                    break
                
                # Initialize U with small random integers
                # Or use a structured initialization
                U = np.random.randint(-2, 3, size=(9, r))
                
                # Solve for M_r
                # T[a, :, :] = sum_r U[a, r] M_r
                # Reshape T to (9, 81) where 81 = 9*9
                T_mat = T.reshape(9, 81)
                
                # We want to find M_r (r x 81) such that T_mat = U @ M
                # If U is 9xr and r < 9, U is not square.
                # We can use least squares: M = pinv(U) @ T_mat
                # But this gives the best fit, not exact.
                # We need exact solution.
                # If U has rank r, and r <= 9, we can check if T_mat is in the column space of U.
                
                # Actually, the equation is T_mat = U M.
                # This is a linear system for M.
                # If U is full column rank, M is unique.
                # M = np.linalg.lstsq(U, T_mat, rcond=None)[0]
                
                try:
                    M = np.linalg.lstsq(U, T_mat, rcond=None)[0]
                except:
                    continue
                
                # Check if the fit is exact
                if not np.allclose(U @ M, T_mat, atol=1e-9):
                    continue
                
                # Now M is (r, 81). Reshape to (r, 9, 9)
                M_3d = M.reshape(r, 9, 9)
                
                # Check if each M_r is rank 1
                valid = True
                V = np.zeros((9, r))
                W = np.zeros((9, r))
                
                for idx in range(r):
                    M_r = M_3d[idx]
                    # Check rank 1
                    # SVD
                    try:
                        U_svd, s, Vt = np.linalg.svd(M_r)
                    except:
                        valid = False
                        break
                    
                    if len(s) < 2 or s[1] > 1e-9:
                        valid = False
                        break
                    
                    # Extract V[:, idx] and W[:, idx]
                    # M_r = V[:, idx] W[:, idx]^T
                    # From SVD: M_r = U_svd[:, 0] * s[0] * Vt[0, :]
                    # So V[:, idx] = U_svd[:, 0] * sqrt(s[0])
                    #    W[:, idx] = Vt[0, :] * sqrt(s[0])
                    # Or just distribute the scalar arbitrarily.
                    # Let's say V[:, idx] = U_svd[:, 0] * s[0]
                    #        W[:, idx] = Vt[0, :]
                    
                    V[:, idx] = U_svd[:, 0] * s[0]
                    W[:, idx] = Vt[0, :]
                
                if not valid:
                    continue
                
                # Verify the full decomposition
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Compute T_recon
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                
                if np.allclose(T_recon, T, atol=1e-9):
                    # We have a valid decomposition!
                    # Check if entries are integer/half-integer
                    # If not, try to round?
                    # The problem allows integer/half-integer.
                    # If they are real numbers, we might need to scale.
                    # But the problem says "exactly decompose".
                    # If the numbers are not nice, we might fail the "integer/half-integer" constraint?
                    # "with integer/half-integer entries"
                    # So we MUST return integer or half-integer entries.
                    
                    # Check if U, V, W are close to integer or half-integer
                    def is_nice(x):
                        # Check if x is close to k/2
                        return np.allclose(x, np.round(x * 2) / 2, atol=1e-9)
                    
                    if is_nice(U) and is_nice(V) and is_nice(W):
                        # Convert to list
                        result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        # This is a valid rank-r decomposition
                        # We want minimal rank, so we return the first one found (lowest r)
                        return result
                    else:
                        # Try to find a scaling factor?
                        # Or just skip and try next random init
                        pass
        
        # If no better decomposition found, return schoolbook
        return get_schoolbook()

# EVOLVE-BLOCK-END
