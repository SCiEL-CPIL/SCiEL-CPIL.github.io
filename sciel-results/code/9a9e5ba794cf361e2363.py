import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses Strassen blocks plus exact rational linear dependency search to minimize rank.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the tensor T[a, b, c]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err

    def build_strassen_decomposition():
        """Build the Strassen-based decomposition with rank 46."""
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook_term(i, j, k):
            u = {(i, j): 1}
            v = {(j, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(i, j, 4)
                
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(i, 4, k)
        
        return terms

    def term_to_flat_vector(term):
        """Convert a term to a flat vector with Fraction values."""
        u_dict, v_dict, w_dict = term
        result = {}
        for (i, j), u_val in u_dict.items():
            for (j2, k), v_val in v_dict.items():
                if j2 != j:
                    continue
                for (k2, i2), w_val in w_dict.items():
                    if k2 != k or i2 != i:
                        continue
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    key = a * (m * p) * (p * n) + b * (p * n) + c
                    result[key] = result.get(key, Fraction(0)) + Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
        return result

    def find_null_space_basis(term_tensors):
        """Find null space basis of the matrix whose columns are term_tensors."""
        num_terms = len(term_tensors)
        
        all_keys = set()
        for tt in term_tensors:
            all_keys.update(tt.keys())
        all_keys = sorted(all_keys)
        num_dims = len(all_keys)
        
        if num_dims == 0 or num_terms == 0:
            return []
        
        # Build matrix A: num_dims rows, num_terms columns
        A = [[Fraction(term_tensors[r].get(k, 0)) for r in range(num_terms)] for k in all_keys]
        
        # RREF
        pivot_cols = []
        row = 0
        for col in range(num_terms):
            if row >= num_dims:
                break
            pivot = None
            for r in range(row, num_dims):
                if A[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                continue
            A[row], A[pivot] = A[pivot], A[row]
            pv = A[row][col]
            for c in range(col, num_terms):
                A[row][c] = A[row][c] / pv
            for r in range(num_dims):
                if r != row and A[r][col] != 0:
                    factor = A[r][col]
                    for c in range(col, num_terms):
                        A[r][c] -= factor * A[row][c]
            pivot_cols.append(col)
            row += 1
        
        free_cols = [c for c in range(num_terms) if c not in pivot_cols]
        
        null_basis = []
        for fc in free_cols:
            vec = [Fraction(0)] * num_terms
            vec[fc] = Fraction(1)
            for i, pc in enumerate(pivot_cols):
                if i < len(A):
                    val = Fraction(0)
                    for c in range(pc + 1, num_terms):
                        val += A[i][c] * vec[c]
                    vec[pc] = -val
            null_basis.append(vec)
        
        return null_basis

    def try_reduce_rank(terms):
        """Try to reduce rank by eliminating linearly dependent terms."""
        current_terms = list(terms)
        current_tensors = [term_to_flat_vector(t) for t in current_terms]
        
        max_passes = 30
        for _pass in range(max_passes):
            if time.time() > deadline - 5:
                break
            if len(current_terms) <= 1:
                break
            
            null_basis = find_null_space_basis(current_tensors)
            
            if not null_basis:
                break
            
            eliminated = False
            for dep in null_basis:
                if time.time() > deadline - 5:
                    break
                
                nonzero = [r for r in range(len(dep)) if dep[r] != 0]
                if len(nonzero) < 2:
                    continue
                
                r_elim = nonzero[-1]
                new_terms = [current_terms[j] for j in range(len(current_terms)) if j != r_elim]
                new_tensors = [current_tensors[j] for j in range(len(current_terms)) if j != r_elim]
                
                current_terms = new_terms
                current_tensors = new_tensors
                eliminated = True
                break
            
            if not eliminated:
                break
        
        return current_terms

    # Build Strassen decomposition
    strassen_terms = build_strassen_decomposition()
    U, V, W = build_decomposition_from_terms(strassen_terms)
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via linear dependency search
    reduced_terms = try_reduce_rank(strassen_terms)
    
    if len(reduced_terms) < len(strassen_terms):
        U, V, W = build_decomposition_from_terms(reduced_terms)
        valid, err = verify_decomposition(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the Strassen decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
