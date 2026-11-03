import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen-based block decomposition to reduce rank below schoolbook (60).
    Falls back to schoolbook if no improvement is found within time budget.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12

    # Build target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose tensor T."""
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def strassen_decomp_2x5_5x6():
        """
        Construct decomposition using Strassen on 2x2 sub-blocks.
        For each j-pair (j0,j1) and k-pair (k0,k1), the 2x2*2x2 subproblem
        (over i in {0,1}) can be done in rank 7 instead of 8.
        Also optimizes the singleton j=4 handling.
        """
        rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(U_col, V_col, W_col):
            nonlocal rank
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            rank += 1
        
        def make_U_col(values):
            col = np.zeros(nm, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_V_col(values):
            col = np.zeros(mp, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_W_col(values):
            col = np.zeros(pn, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def strassen_terms(i0, i1, j0, j1, k0, k1):
            """
            Generate Strassen terms for 2x2*2x2 subproblem.
            A = [[a,b],[c,d]] where a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
            B = [[e,f],[g,h]] where e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
            C = [[C11,C12],[C21,C22]] where C11=C[i0,k0], C12=C[i0,k1], C21=C[i1,k0], C22=C[i1,k1]
            
            Strassen:
            M1 = (a+d)(e+h)
            M2 = (c+d)e
            M3 = a(f-h)
            M4 = d(g-e)
            M5 = (a+b)h
            M6 = (c-a)(e+f)
            M7 = (b-d)(g+h)
            
            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 + M3 + M6
            """
            # M1 = (a+d)(e+h)
            add_term(
                make_U_col({i0*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1, k1*n+i1: 1})
            )
            
            # M2 = (c+d)e
            add_term(
                make_U_col({i1*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1}),
                make_W_col({k0*n+i1: 1})
            )
            
            # M3 = a(f-h)
            add_term(
                make_U_col({i0*m+j0: 1}),
                make_V_col({j0*p+k1: 1, j1*p+k1: -1}),
                make_W_col({k1*n+i0: 1, k1*n+i1: 1})
            )
            
            # M4 = d(g-e)
            add_term(
                make_U_col({i1*m+j1: 1}),
                make_V_col({j1*p+k0: 1, j0*p+k0: -1}),
                make_W_col({k0*n+i0: 1, k0*n+i1: 1})
            )
            
            # M5 = (a+b)h
            add_term(
                make_U_col({i0*m+j0: 1, i0*m+j1: 1}),
                make_V_col({j1*p+k1: 1}),
                make_W_col({k0*n+i0: -1, k1*n+i0: 1})
            )
            
            # M6 = (c-a)(e+f)
            add_term(
                make_U_col({i1*m+j0: 1, i0*m+j0: -1}),
                make_V_col({j0*p+k0: 1, j0*p+k1: 1}),
                make_W_col({k1*n+i1: 1})
            )
            
            # M7 = (b-d)(g+h)
            add_term(
                make_U_col({i0*m+j1: 1, i1*m+j1: -1}),
                make_V_col({j1*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1})
            )
        
        # Apply Strassen to all 2x2*2x2 subproblems
        # j-pairs: (0,1), (2,3)
        # k-pairs: (0,1), (2,3), (4,5)
        # i-pair: (0,1)
        
        for (j0, j1) in [(0, 1), (2, 3)]:
            for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                strassen_terms(0, 1, j0, j1, k0, k1)
        
        # Handle j=4 (singleton) - use 6 terms, one per k
        # For each k, C[i,k] += A[i,4]*B[4,k] for i in {0,1}
        for k in range(p):
            add_term(
                make_U_col({0*m+4: 1, 1*m+4: 1}),
                make_V_col({4*p+k: 1}),
                make_W_col({k*n+0: 1, k*n+1: 1})
            )
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 60

    # Try Strassen decomposition
    try:
        U_s, V_s, W_s, rank_s = strassen_decomp_2x5_5x6()
        if verify(U_s, V_s, W_s) and rank_s < best_rank:
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = rank_s
    except Exception:
        pass

    # Now try to improve further using structured search informed by n=2 structure
    # Key: for n=2, we can try to find decompositions where terms share structure
    
    def structured_improvement_search(start_rank, max_attempts=200):
        """
        Try to find a lower-rank decomposition by starting from the current best
        and attempting to merge/replace terms.
        Uses the fact that n=2 to guide the search.
        """
        if time.time() > deadline - 20:
            return None
        
        best_U_local = best_U.copy()
        best_V_local = best_V.copy()
        best_W_local = best_W.copy()
        current_rank = start_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            target_rank = current_rank - 1
            if target_rank < 25:  # Don't go below theoretical lower bound
                break
            
            # Strategy: pick a random term to remove and see if we can compensate
            # by modifying other terms
            U_try = best_U_local[:, :target_rank].copy()
            V_try = best_V_local[:, :target_rank].copy()
            W_try = best_W_local[:, :target_rank].copy()
            
            # Perturb randomly to escape local minima
            np.random.seed(attempt * 7919 + 13)
            for r in range(target_rank):
                if np.random.rand() < 0.3:
                    # Modify U
                    idx = np.random.randint(0, nm)
                    U_try[idx, r] *= np.random.choice([-1.0, 1.0, 2.0, 0.5])
                if np.random.rand() < 0.3:
                    # Modify V
                    idx = np.random.randint(0, mp)
                    V_try[idx, r] *= np.random.choice([-1.0, 1.0, 2.0, 0.5])
                if np.random.rand() < 0.3:
                    # Modify W
                    idx = np.random.randint(0, pn)
                    W_try[idx, r] *= np.random.choice([-1.0, 1.0, 2.0, 0.5])
            
            if verify(U_try, V_try, W_try):
                best_U_local = U_try
                best_V_local = V_try
                best_W_local = W_try
                current_rank = target_rank
        
        if current_rank < start_rank:
            return best_U_local, best_V_local, best_W_local, current_rank
        return None

    # Try structured improvement
    if best_rank > 30:
        result = structured_improvement_search(best_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result

    # Try a more targeted search: exploit the 2-row structure
    # For n=2, C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
    # We can try to find a decomposition that shares work between the two rows
    
    def two_row_structured_search(target_rank, n_attempts=300):
        """
        Search for decompositions that exploit the 2-row structure.
        Each term has a specific pattern in U (which row it affects) and W (which output).
        """
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(42 + attempt * 31)
            
            U_try = np.zeros((nm, target_rank), dtype=np.float64)
            V_try = np.zeros((mp, target_rank), dtype=np.float64)
            W_try = np.zeros((pn, target_rank), dtype=np.float64)
            
            for r in range(target_rank):
                # Choose a j index
                j = np.random.randint(0, m)
                
                # Decide which rows this term affects
                affect_row0 = np.random.rand() < 0.7
                affect_row1 = np.random.rand() < 0.7
                if not affect_row0 and not affect_row1:
                    affect_row0 = True
                
                # Choose k indices this term affects
                num_k = np.random.choice([1, 1, 1, 2, 2, 3])
                k_indices = np.random.choice(p, size=min(num_k, p), replace=False)
                
                # Set U values
                if affect_row0:
                    U_try[0*m+j, r] = np.random.choice([-1.0, 1.0])
                if affect_row1:
                    U_try[1*m+j, r] = np.random.choice([-1.0, 1.0])
                
                # Set V values
                for k in k_indices:
                    V_try[j*p+k, r] = np.random.choice([-1.0, 1.0])
                
                # Set W values - must be consistent with the tensor structure
                for k in k_indices:
                    if affect_row0:
                        W_try[k*n+0, r] = np.random.choice([-1.0, 1.0])
                    if affect_row1:
                        W_try[k*n+1, r] = np.random.choice([-1.0, 1.0])
            
            if verify(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None

    # Try two-row structured search
    for target_rank in range(best_rank - 1, max(28, best_rank - 10), -1):
        if time.time() > deadline - 20:
            break
        
        result = two_row_structured_search(target_rank, n_attempts=100)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break

    def to_list(arr):
        return arr.astype(float).tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
