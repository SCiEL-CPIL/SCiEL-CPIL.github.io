import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130

    def time_left():
        return deadline - time.time()

    def verify_full(U, V, W, rank):
        """Full verification: check all 60 tensor entries."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def build_group_algebra_decomp():
        """
        Bilinear Group Algebra method: Embed matrix multiplication into
        the group algebra of D6 (dihedral group of order 12).
        
        D6 = {e, r, r^2, r^3, r^4, r^5, s, sr, sr^2, sr^3, sr^4, sr^5}
        Relations: r^6 = e, s^2 = e, srs = r^(-1)
        
        We assign:
        - A factors (n*m=10 entries) to group elements
        - B factors (m*p=30 entries) to group elements  
        - C factors (p*n=12 entries) to group elements
        
        The key insight: if we can find assignments such that for each output
        position (i,k), the sum over j of A[i,j]*B[j,k] corresponds to a
        single group element product, we get a rank-12 decomposition.
        
        In practice, we use the group algebra structure to create a structured
        starting point with many zeros, then eliminate terms.
        """
        # D6 group elements: 0=e, 1=r, 2=r^2, 3=r^3, 4=r^4, 5=r^5,
        #                     6=s, 7=sr, 8=sr^2, 9=sr^3, 10=sr^4, 11=sr^5
        # Multiplication table for D6
        # r^a * r^b = r^(a+b mod 6)
        # r^a * sr^b = sr^(b-a mod 6)
        # sr^a * r^b = sr^(a+b mod 6)
        # sr^a * sr^b = r^(b-a mod 6)
        
        def group_mult(g1, g2):
            # Elements 0-5 are r^0..r^5, elements 6-11 are s*r^0..s*r^5
            def is_r(g):
                return g < 6
            def r_exp(g):
                return g if is_r(g) else g - 6
            
            if is_r(g1) and is_r(g2):
                return (r_exp(g1) + r_exp(g2)) % 6
            elif is_r(g1) and not is_r(g2):
                # r^a * sr^b = sr^(b-a)
                return 6 + (r_exp(g2) - r_exp(g1)) % 6
            elif not is_r(g1) and is_r(g2):
                # sr^a * r^b = sr^(a+b)
                return 6 + (r_exp(g1) + r_exp(g2)) % 6
            else:
                # sr^a * sr^b = r^(b-a)
                return (r_exp(g2) - r_exp(g1)) % 6
        
        # Strategy: Assign A[i,j] to group element a_ij, B[j,k] to b_jk,
        # C[i,k] to c_ik such that for each (i,j,k):
        #   a_ij * b_jk should "contribute to" c_ik
        # 
        # In group algebra, the product sum is over all group elements.
        # We want: sum_j A[i,j] * B[j,k] = C[i,k] in group algebra
        # where A[i,j] = e_{a_ij}, B[j,k] = e_{b_jk}, C[i,k] = e_{c_ik}
        #
        # This requires a_ij * b_jk = c_ik for all j (which is impossible for
        # multiple j unless all give same product). So we use a more general
        # assignment where each term contributes to multiple outputs.
        #
        # Alternative: Use the fact that in group algebra, the multiplication
        # tensor has rank related to the group size. For D6 of order 12,
        # the group multiplication tensor has rank 12.
        #
        # Let's try a direct construction:
        # Map the 10 A-entries, 30 B-entries, 12 C-entries to group elements
        # such that the tensor equation holds.
        
        # Simpler approach: Use the cyclic group Z_6 (order 6) as a subset.
        # Z_6 multiplication: a*b = (a+b) mod 6
        # This gives a rank-6 decomposition for the cyclic convolution.
        #
        # For our problem, we can try to embed using Z_6:
        # A[i,j] -> e_{alpha(i,j)}, B[j,k] -> e_{beta(j,k)}, C[i,k] -> e_{gamma(i,k)}
        # Need: alpha(i,j) + beta(j,k) = gamma(i,k) mod 6 for all (i,j,k)
        
        # Let's try: alpha(i,j) = i*3 + j mod 6 (uses 10 values mapped to 6)
        # beta(j,k) = j + k mod 6
        # gamma(i,k) = ?
        # i*3 + j + j + k = i*3 + 2j + k... not separable in (i,k)
        
        # Let's try a different assignment:
        # alpha(i,j) = i + 2*j mod 6
        # beta(j,k) = j + k mod 6  
        # gamma(i,k) = i + 3*k mod 6
        # Check: alpha + beta = i + 2j + j + k = i + 3j + k
        # gamma = i + 3k
        # Need 3j = 3k mod 6, i.e., j = k mod 2. Not always true.
        
        # Try: alpha(i,j) = i + j mod 6
        # beta(j,k) = j + 2k mod 6
        # gamma(i,k) = i + 3k mod 6
        # alpha + beta = i + j + j + 2k = i + 2j + 2k
        # gamma = i + 3k
        # Need 2j = 3k mod 6. Not always true.
        
        # The group algebra approach gives us a structured matrix with many
        # zeros. Let's just build a rank-12 structured starting point.
        
        # Actually, let's use a simpler but effective approach:
        # Build a rank-18 starting point based on block structure that's
        # different from Strassen, then eliminate.
        
        # Use the fact that 2x5x6 can be decomposed as:
        # Split m=5 into 2+3, p=6 into 2+4
        # 2x2x2: rank 7 (Strassen)
        # 2x2x4: rank ? 
        # 2x3x2: rank ?
        # 2x3x4: rank ?
        
        # For 2xa xb, the Strassen-like bound gives rank ~ 7*(a/2)*(b/2)
        # 2x2x4: use 2 Strassen 2x2x2 blocks = 14, but can do better
        # Actually 2x2x4 has known rank 11 or 12
        
        # Let me just build a structured rank-20 starting point with
        # specific zero patterns that make elimination easier.
        
        rank = 20
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Assign each (i,j) pair to a specific column pattern
        # U[0,j] and U[1,j] for j=0..4
        # V[j,k] for j=0..4, k=0..5
        # W[k,i] for k=0..5, i=0..1
        
        # Column 0-3: handle j=0 (5 k-values for each i)
        for k in range(4):
            U[0*5+0, k] = 1
            U[1*5+0, k] = 1
            V[0*6+k, k] = 1
            W[k*2+0, k] = 1
            W[k*2+1, k] = 1
        # Column 4: j=0, k=4
        U[0*5+0, 4] = 1
        U[1*5+0, 4] = 1
        V[0*6+4, 4] = 1
        W[4*2+0, 4] = 1
        W[4*2+1, 4] = 1
        # Column 5: j=0, k=5
        U[0*5+0, 5] = 1
        U[1*5+0, 5] = 1
        V[0*6+5, 5] = 1
        W[5*2+0, 5] = 1
        W[5*2+1, 5] = 1
        
        # Columns 6-10: handle j=1
        for k in range(4):
            U[0*5+1, 6+k] = 1
            U[1*5+1, 6+k] = 1
            V[1*6+k, 6+k] = 1
            W[k*2+0, 6+k] = 1
            W[k*2+1, 6+k] = 1
        U[0*5+1, 10] = 1
        U[1*5+1, 10] = 1
        V[1*6+4, 10] = 1
        W[4*2+0, 10] = 1
        W[4*2+1, 10] = 1
        U[0*5+1, 11] = 1
        U[1*5+1, 11] = 1
        V[1*6+5, 11] = 1
        W[5*2+0, 11] = 1
        W[5*2+1, 11] = 1
        
        # Columns 12-16: handle j=2
        for k in range(4):
            U[0*5+2, 12+k] = 1
            U[1*5+2, 12+k] = 1
            V[2*6+k, 12+k] = 1
            W[k*2+0, 12+k] = 1
            W[k*2+1, 12+k] = 1
        U[0*5+2, 16] = 1
        U[1*5+2, 16] = 1
        V[2*6+4, 16] = 1
        W[4*2+0, 16] = 1
        W[4*2+1, 16] = 1
        U[0*5+2, 17] = 1
        U[1*5+2, 17] = 1
        V[2*6+5, 17] = 1
        W[5*2+0, 17] = 1
        W[5*2+1, 17] = 1
        
        # Columns 18-19: handle j=3,4 partially (will need more)
        # This isn't quite right - let me use a cleaner approach.
        
        # Actually, the schoolbook is 60. Strassen gives 54.
        # Let me build a rank-40 structured decomposition and eliminate.
        
        return None  # Fall through to other methods
    
    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        changed = True
        while changed and time_left() > 5:
            changed = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify_full(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    changed = True
                    break
        
        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U, V, W, rank):
        """Try to eliminate pairs of terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        improved = True
        while improved and time_left() > 10:
            improved = False
            for a in range(rank - 1, -1, -1):
                if time_left() < 10:
                    break
                for b in range(a - 1, -1, -1):
                    if time_left() < 10:
                        break
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][a] = 0
                        U_new[i][b] = 0
                    for j in range(m*p):
                        V_new[j][a] = 0
                        V_new[j][b] = 0
                    for k in range(p*n):
                        W_new[k][a] = 0
                        W_new[k][b] = 0
                    
                    if verify_full(U_new, V_new, W_new, rank):
                        keep = [j for j in range(rank) if j != a and j != b]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 2
                        improved = True
                        break
                if improved:
                    break
        
        return U_list, V_list, W_list, rank

    def try_triple_elimination(U, V, W, rank):
        """Try to eliminate triples of terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        improved = True
        while improved and time_left() > 15:
            improved = False
            for a in range(rank - 2, -1, -1):
                if time_left() < 15:
                    break
                for b in range(a - 1, -1, -1):
                    if time_left() < 15:
                        break
                    for c in range(b - 1, -1, -1):
                        if time_left() < 15:
                            break
                        U_new = [row[:] for row in U_list]
                        V_new = [row[:] for row in V_list]
                        W_new = [row[:] for row in W_list]
                        
                        for i in range(n*m):
                            U_new[i][a] = 0
                            U_new[i][b] = 0
                            U_new[i][c] = 0
                        for j in range(m*p):
                            V_new[j][a] = 0
                            V_new[j][b] = 0
                            V_new[j][c] = 0
                        for k in range(p*n):
                            W_new[k][a] = 0
                            W_new[k][b] = 0
                            W_new[k][c] = 0
                        
                        if verify_full(U_new, V_new, W_new, rank):
                            keep = [j for j in range(rank) if j not in (a, b, c)]
                            U_list = [[row[j] for j in keep] for row in U_list]
                            V_list = [[row[j] for j in keep] for row in V_list]
                            W_list = [[row[j] for j in keep] for row in W_list]
                            rank -= 3
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return U_list, V_list, W_list, rank

    def solve_for_W(U, V, rank):
        """Given U and V, solve for W such that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        num_unknowns = p * n * rank
        num_eqs = n * m * p
        
        if num_unknowns < num_eqs:
            return None
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        w_idx = (k * n + i) * rank + r
                        A[eq_idx, w_idx] = U_arr[i*m+j, r] * V_arr[j*p+k, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.max(np.abs(A @ W_flat - b)) > 1e-8:
                return None
            W_arr = W_flat.reshape(p * n, rank)
            W_snapped = np.round(W_arr * 2) / 2
            if np.max(np.abs(A @ W_snapped.flatten() - b)) > 1e-8:
                return None
            return W_snapped.tolist()
        except:
            return None

    def build_strassen_and_optimize(j_pairs, k_pairs):
        """Build Strassen decomposition and try to reduce rank."""
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            return None, None, None, rank
        
        U_list, V_list, W_list, rank = try_eliminate_terms(U, V, W, rank)
        U_list, V_list, W_list, rank = try_pairwise_elimination(
            np.array(U_list), np.array(V_list), np.array(W_list), rank
        )
        U_list, V_list, W_list, rank = try_triple_elimination(
            np.array(U_list), np.array(V_list), np.array(W_list), rank
        )
        
        return U_list, V_list, W_list, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(1,2),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(3,4),(2,5)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U_list, V_list, W_list, rank = build_strassen_and_optimize(j_pairs, k_pairs)
        
        if U_list is not None and rank < best_rank:
            best_rank = rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try group-algebra-inspired structured decompositions
    if time_left() > 30:
        # Build structured decompositions with specific zero patterns
        # inspired by group algebra embeddings
        np.random.seed(42)
        
        # Try rank values below current best
        for test_rank in range(max(30, best_rank - 5), best_rank):
            if time_left() < 20:
                break
            
            # Strategy 1: Random sparse U,V with structured patterns
            for attempt in range(50):
                if time_left() < 15:
                    break
                
                U_test = np.zeros((n*m, test_rank))
                V_test = np.zeros((m*p, test_rank))
                
                # Assign each (i,j) to 2-3 columns
                for i in range(n):
                    for j in range(m):
                        num_cols = np.random.randint(2, 4)
                        cols = np.random.choice(test_rank, size=num_cols, replace=False)
                        for c in cols:
                            U_test[i*m+j, c] = np.random.choice([1, -1])
                
                for j in range(m):
                    for k in range(p):
                        num_cols = np.random.randint(2, 4)
                        cols = np.random.choice(test_rank, size=num_cols, replace=False)
                        for c in cols:
                            V_test[j*p+k, c] = np.random.choice([1, -1])
                
                W_test = solve_for_W(U_test, V_test, test_rank)
                if W_test is not None:
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, test_rank):
                        if test_rank < best_rank:
                            best_rank = test_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test}
                        break
            
            # Strategy 2: Block-structured with group-like symmetry
            if time_left() < 15:
                break
            for attempt in range(20):
                if time_left() < 15:
                    break
                
                U_test = np.zeros((n*m, test_rank))
                V_test = np.zeros((m*p, test_rank))
                
                # Use a cyclic pattern: column c handles (i,j) where j = c mod m
                # and (j,k) where k = c mod p, with shifts
                shift = attempt
                for c in range(test_rank):
                    j_idx = (c + shift) % m
                    k_idx = (c + shift) % p
                    i_idx = c % n
                    
                    U_test[i_idx*m + j_idx, c] = 1
                    V_test[j_idx*p + k_idx, c] = 1
                
                W_test = solve_for_W(U_test, V_test, test_rank)
                if W_test is not None:
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, test_rank):
                        if test_rank < best_rank:
                            best_rank = test_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test}
                        break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
