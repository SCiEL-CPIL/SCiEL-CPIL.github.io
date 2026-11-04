"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k]."""
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            j_a2 = j // p
            k_b = j % p
            if j_a != j_a2:
                continue
            for k in range(w_dim):
                k_b2 = k // n
                i_c = k % n
                if i_a != i_c or k_b != k_b2:
                    continue
                T[i, j, k] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if the decomposition exactly reconstructs T."""
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.max(np.abs(recon - T)) < tol


def _try_strassen_rank17():
    """
    Construct a rank-17 decomposition using Strassen's algorithm on the 2x2 block
    (columns 0,1 of the middle dimension) and direct multiplication for column 2.
    
    The 2x3 matrix A has columns A[:,0], A[:,1], A[:,2].
    The 3x3 matrix B has rows B[0,:], B[1,:], B[2,:].
    
    Product C = A * B has entries:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We split: C[:,0:2] uses Strassen on 2x2 x 2x2, and C[:,2] uses direct 2x1 x 1x3.
    
    Strassen for 2x2 * 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For our case, the "2x2" block uses A columns 0,1 and B rows 0,1.
    The "2x1 x 1x3" part uses A column 2 and B row 2.
    
    Total rank: 7 (Strassen) + 2 (direct for col 2) = 9... wait that's not right.
    
    Actually, let me reconsider. The full computation is:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    For k=0,1: we can use Strassen on the 2x2 subproblem (A[:,0:2] * B[0:2,:])
    For k=2: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    
    Hmm, this doesn't cleanly separate. Let me think differently.
    
    Actually, the standard approach: treat the problem as multiplying a 2x3 by 3x3.
    We can view this as:
    - The first two columns of B (3x2) multiplied by A (2x3) gives a 2x2 result
    - The third column of B (3x1) multiplied by A (2x3) gives a 2x1 result
    
    But Strassen applies to square matrices. Let me use a different decomposition.
    
    Alternative: Use the fact that M_{2,3,3} can be decomposed by considering
    the 2x2 sub-block and the remaining terms.
    
    Let's use a known rank-17 construction. The key idea:
    - Use 7 Strassen products for the 2x2 x 2x2 part (columns 0,1 of A, rows 0,1 of B)
    - Use additional products to handle the cross terms involving column 2 of A and row 2 of B
    
    Actually, let me just construct it directly by solving the linear system.
    """
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    T = _build_tensor()
    
    rank = 17
    U = np.zeros((u_dim, rank))
    V = np.zeros((v_dim, rank))
    W = np.zeros((w_dim, rank))
    
    # Index conventions:
    # U index i = i_a * m + j_a, where i_a in {0,1} (row of A), j_a in {0,1,2} (col of A)
    # V index j = j_a * p + k_b, where j_a in {0,1,2} (row of B), k_b in {0,1,2} (col of B)
    # W index k = k_b * n + i_a, where k_b in {0,1,2} (col of B), i_a in {0,1} (row of C)
    
    # Strassen products for the 2x2 block (A cols 0,1; B rows 0,1):
    # These compute C[i,0] and C[i,1] for i in {0,1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # In our indexing:
    # U entries: A[0,0] is U[0*3+0, c] = U[0, c], A[1,1] is U[1*3+1, c] = U[4, c]
    # V entries: B[0,0] is V[0*3+0, c] = V[0, c], B[1,0] is V[1*3+0, c] = V[3, c]
    # W entries: result goes to C[0,0] = W[0*2+0, c] = W[0, c] and C[1,0] = W[0*2+1, c] = W[1, c]
    
    # Let me define the 7 Strassen terms:
    # Term 0: M1 = (A00+A11)*(B00+B10) -> contributes to C00, C10
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1  # C[0,0] and C[1,0]
    
    # Term 1: M2 = (A10+A11)*B00 -> contributes to C10, C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1  # C[1,0] and C[1,1]
    
    # Term 2: M3 = A00*(B01-B11) -> contributes to C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[3, 2] = 1; W[5, 2] = 1  # C[0,1] and C[1,1]
    
    # Term 3: M4 = A11*(B10-B00) -> contributes to C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # Term 4: M5 = (A00+A01)*B11 -> contributes to C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1  # C[0,0] and C[0,1]
    
    # Term 5: M6 = (A10-A00)*(B00+B01) -> contributes to C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[5, 5] = 1  # C[1,1]
    
    # Term 6: M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1; W[5, 6] = 1  # C[0,0] and C[1,1]
    
    # Now we need to handle the terms involving A[:,2] and B[2,:]:
    # C[i,k] includes A[i,2]*B[2,k]
    # This is a 2x1 * 1x3 = 2x3 subproblem, which has rank 2*1*3 = 6 if done naively.
    # But we only need rank 17 total, so we have 17-7 = 10 slots for this part.
    # Actually, we need to also handle cross terms.
    
    # Wait, I think I'm overcomplicating this. Let me reconsider.
    
    # The Strassen terms above compute:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # 
    # But these only account for the contributions from A[:,0:2] * B[0:2,:].
    # We still need:
    # C[i,0] += A[i,2] * B[2,0]
    # C[i,1] += A[i,2] * B[2,1]
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    
    # Hmm, this approach doesn't cleanly separate. Let me try a different strategy.
    
    # Alternative: Use a rank-17 construction based on known results.
    # The tensor M_{2,3,3} has known rank 17 (it's a well-known result).
    # Let me construct it using a linear algebra approach.
    
    # Actually, let me try a simpler approach: use the schoolbook rank-18 and
    # try to reduce it by 1 using a clever identity.
    
    # For now, let me just return the schoolbook rank-18 as a fallback,
    # and try to find rank 17 through search.
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    Returns dict with U (6 x r), V (9 x r), W (6 x r) where r is the rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    T = _build_tensor()
    
    # Schoolbook decomposition as baseline (rank 18)
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    def check_exact(U_r, V_r, W_r, tol=1e-9):
        """Check if the decomposition is exact."""
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.max(np.abs(recon - T)) < tol
    
    # Try to find rank-17 decomposition using structured search
    def try_rank17(max_time=60.0):
        """Search for a rank-17 decomposition."""
        search_start = time.time()
        rank = 17
        
        # Strategy: start with schoolbook rank-18 and try to merge two terms.
        # The key is to find two schoolbook terms that can be combined.
        
        # Schoolbook terms are indexed by (i_a, j_a, k_b) where:
        # i_a in {0,1}, j_a in {0,1,2}, k_b in {0,1,2}
        # Term (i_a, j_a, k_b) has:
        #   U[i_a*m + j_a, c] = 1
        #   V[j_a*p + k_b, c] = 1
        #   W[k_b*n + i_a, c] = 1
        
        # To merge two terms, we need to find a linear combination that
        # produces the same effect with one fewer rank-1 term.
        
        # Let's try: take two terms and see if their "overlap" allows merging.
        # For terms (i_a1, j_a1, k_b1) and (i_a2, j_a2, k_b2):
        # They share a U entry if i_a1*m+j_a1 == i_a2*m+j_a2
        # They share a V entry if j_a1*p+k_b1 == j_a2*p+k_b2
        # They share a W entry if k_b1*n+i_a1 == k_b2*n+i_a2
        
        # If they share all three, they're the same term.
        # If they share two, we might be able to merge.
        
        # Let's try all pairs and see if we can create a valid rank-17 decomposition.
        
        np.random.seed(42)
        
        # Try random modifications of the schoolbook
        for trial in range(5000):
            if time.time() - search_start > max_time:
                break
            
            U_try = U_s[:, :17].copy()
            V_try = V_s[:, :17].copy()
            W_try = W_s[:, :17].copy()
            
            # Randomly modify a few entries to try to absorb the 18th term
            num_mods = np.random.randint(1, 5)
            for _ in range(num_mods):
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(u_dim)
                    col = np.random.randint(rank)
                    U_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
                elif mat_choice == 1:
                    row = np.random.randint(v_dim)
                    col = np.random.randint(rank)
                    V_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
                else:
                    row = np.random.randint(w_dim)
                    col = np.random.randint(rank)
                    W_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
            
            if check_exact(U_try, V_try, W_try, tol=1e-9):
                return U_try, V_try, W_try
        
        return None
    
    # Try rank 17
    if time.time() < deadline - 20:
        result = try_rank17(max_time=min(80, deadline - time.time() - 10))
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = 17
    
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
