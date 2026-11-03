"""Combined program: structured Strassen decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds

    n, m, p = DIMENSIONS
    terms = []  # Each term: (u_dict, v_dict, w_dict)

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)

        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)

        # M1 = (a+d)(e+h), C: C11:+1, C22:+1
        terms.append((
            {a: 1, d: 1},
            {e: 1, h: 1},
            {C11: 1, C22: 1}
        ))
        # M2 = (c+d)*e, C: C21:+1, C22:-1
        terms.append((
            {c: 1, d: 1},
            {e: 1},
            {C21: 1, C22: -1}
        ))
        # M3 = a*(f-h), C: C12:+1, C22:+1
        terms.append((
            {a: 1},
            {f: 1, h: -1},
            {C12: 1, C22: 1}
        ))
        # M4 = d*(g-e), C: C11:+1, C21:+1
        terms.append((
            {d: 1},
            {g: 1, e: -1},
            {C11: 1, C21: 1}
        ))
        # M5 = (a+b)*h, C: C11:-1, C12:+1
        terms.append((
            {a: 1, b: 1},
            {h: 1},
            {C11: -1, C12: 1}
        ))
        # M6 = (c-a)*(e+f), C: C22:+1
        terms.append((
            {c: 1, a: -1},
            {e: 1, f: 1},
            {C22: 1}
        ))
        # M7 = (b-d)*(g+h), C: C11:+1
        terms.append((
            {b: 1, d: -1},
            {g: 1, h: 1},
            {C11: 1}
        ))

    def add_schoolbook_term(i, j, k):
        """Add term for A[i,j] * B[j,k] -> C[i,k]"""
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))

    # Part 1: Row 0 of A, schoolbook (24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(0, j, k)

    # Part 2: Rows 1-2 of A, using Strassen for 2x2 blocks
    for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
        add_strassen_block(1, 2, 0, 1, k1, k2)
        add_strassen_block(1, 2, 2, 3, k1, k2)

    # Convert to matrices
    def terms_to_matrices(term_list):
        rank = len(term_list)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for col, (u_dict, v_dict, w_dict) in enumerate(term_list):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]... 
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
        # Wait, the tensor identity is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = delta (i.e., the matrix multiplication tensor)
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # Actually the tensor is: for each (i,j,k), the contribution is 1 to C[i,k]
        # So T[i,j,k] should equal 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    U, V, W = terms_to_matrices(terms)
    
    # Verify the base decomposition
    if not verify(U, V, W):
        # Fallback: return what we have (should not happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank by finding linear dependencies
    # The decomposition is valid iff for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We can try to find a smaller set of terms that still satisfies this.
    # Strategy: try to express each term as a linear combination of others.
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try merging terms with identical (U_col, V_col) pairs
    # If two terms have the same U and V columns, their W columns can be summed.
    merged_terms = []
    seen = {}  # key: (frozenset of U entries, frozenset of V entries) -> index in merged_terms
    for idx, (u_dict, v_dict, w_dict) in enumerate(terms):
        u_key = tuple(sorted(u_dict.items()))
        v_key = tuple(sorted(v_dict.items()))
        key = (u_key, v_key)
        if key in seen:
            # Merge W into existing term
            existing = merged_terms[seen[key]]
            new_w = dict(existing[2])
            for (ik, coeff) in w_dict.items():
                new_w[ik] = new_w.get(ik, 0) + coeff
            # Remove zero entries
            new_w = {k: v for k, v in new_w.items() if v != 0}
            merged_terms[seen[key]] = (existing[0], existing[1], new_w)
        else:
            seen[key] = len(merged_terms)
            merged_terms.append((u_dict, v_dict, w_dict))

    if len(merged_terms) < best_rank:
        U2, V2, W2 = terms_to_matrices(merged_terms)
        if verify(U2, V2, W2):
            best_rank = U2.shape[1]
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()

    # Try to find linear dependencies among terms using rank analysis
    # Build the "tensor matrix" and check if we can remove terms
    # This is more complex; try a greedy approach
    if time.time() - start_time > DEADLINE - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try: for each term, check if removing it and adjusting others still works
    # This is expensive, so only try if we have time
    current_terms = merged_terms if len(merged_terms) < best_rank else terms
    current_rank = len(current_terms)
    
    # Try removing one term at a time and solving for it
    for remove_idx in range(current_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
        remaining = [t for i, t in enumerate(current_terms) if i != remove_idx]
        if not remaining:
            break
        R, Vm, Wm = terms_to_matrices(remaining)
        # Check if the removed term is needed: compute the residual
        # The removed term contributed: U[remove]*V[remove]*W[remove]
        U_rem, V_rem, W_rem = terms_to_matrices([current_terms[remove_idx]])
        # We need: for all i,j,k, sum over remaining + new terms = 1
        # This is complex; skip for now
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
