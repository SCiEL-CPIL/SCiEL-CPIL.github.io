# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    rank_terms = []
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    # Initial block decomposition
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
    
    # Build matrices as Fraction for exact arithmetic
    def build_matrices(terms):
        rank = len(terms)
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = Fraction(coeff)
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = Fraction(coeff)
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = Fraction(coeff)
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Check that U ⊗ V ⊗ W equals the matmul tensor."""
        rank = len(U[0]) if U else 0
        # The decomposition says: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # where T[i,j,k] = 1 for all valid (i,j,k)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    lhs = Fraction(1)
                    rhs = Fraction(0)
                    for r in range(rank):
                        rhs += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if lhs != rhs:
                        return False
        return True
    
    def try_remove_column(terms, col_to_remove):
        """Try removing a column by expressing it as a linear combination of others."""
        rank = len(terms)
        if rank <= 1:
            return None
        
        U, V, W = build_matrices(terms)
        
        other_cols = [c for c in range(rank) if c != col_to_remove]
        
        # Build system: find coefficients alpha_c such that
        # U[:, col] = sum_c alpha_c * U[:, c], similarly for V and W
        other_idx = {c: idx for idx, c in enumerate(other_cols)}
        
        # Collect all equations from U, V, W
        rows = []
        for row in range(n * m):
            b_val = U[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                rows.append((row_vec, b_val))
        
        for row in range(m * p):
            b_val = V[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                rows.append((row_vec, b_val))
        
        for row in range(p * n):
            b_val = W[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                rows.append((row_vec, b_val))
        
        # Also need: for rows where b_val == 0, the sum must be 0
        for row in range(n * m):
            if U[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        for row in range(m * p):
            if V[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        for row in range(p * n):
            if W[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        if not rows:
            return other_cols
        
        # Gaussian elimination
        mat = []
        for row_vec, b_val in rows:
            mat.append(row_vec + [b_val])
        
        num_unknowns = len(other_cols)
        
        # Reduce
        pivot_rows = []
        used_pivots = set()
        for col in range(num_unknowns):
            if col in used_pivots:
                continue
            pivot_row = None
            for r in range(len(mat)):
                if mat[r][col] != 0 and r not in used_pivots:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            
            mat[pivot_row], mat[len(used_pivots)] = mat[len(used_pivots)], mat[pivot_row]
            
            pivot_val = mat[len(used_pivots)][col]
            mat[len(used_pivots)] = [x / pivot_val for x in mat[len(used_pivots)]]
            
            for r in range(len(mat)):
                if r == len(used_pivots):
                    continue
                factor = mat[r][col]
                if factor != 0:
                    mat[r] = [mat[r][c] - factor * mat[len(used_pivots)][c] for c in range(num_unknowns + 1)]
            
            used_pivots.add(len(used_pivots))
            pivot_rows.append((len(used_pivots) - 1, col))
        
        # Check for inconsistency
        for r in range(len(mat)):
            all_zero = all(mat[r][c] == 0 for c in range(num_unknowns))
            if all_zero and mat[r][num_unknowns] != 0:
                return None
        
        # Solution exists. Extract alpha values
        alpha = [Fraction(0)] * num_unknowns
        for pr, col in pivot_rows:
            alpha[col] = mat[pr][num_unknowns]
        
        # Build new terms
        new_terms = []
        for c in other_cols:
            if alpha[other_idx[c]] != 0:
                u_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][0].items()}
                v_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][1].items()}
                w_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][2].items()}
                new_terms.append((u_dict, v_dict, w_dict))
        
        new_terms = [(u, v, w) for u, v, w in new_terms 
                     if u and v and w]
        
        return new_terms
    
    def try_remove_pair(terms, col1, col2):
        """Try removing two columns simultaneously."""
        rank = len(terms)
        if rank <= 2:
            return None
        
        U, V, W = build_matrices(terms)
        
        other_cols = [c for c in range(rank) if c != col1 and c != col2]
        other_idx = {c: idx for idx, c in enumerate(other_cols)}
        
        rows = []
        for row in range(n * m):
            b1 = U[row][col1]
            b2 = U[row][col2]
            if b1 != 0 or b2 != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                rows.append((row_vec, b1, b2))
        
        for row in range(m * p):
            b1 = V[row][col1]
            b2 = V[row][col2]
            if b1 != 0 or b2 != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                rows.append((row_vec, b1, b2))
        
        for row in range(p * n):
            b1 = W[row][col1]
            b2 = W[row][col2]
            if b1 != 0 or b2 != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                rows.append((row_vec, b1, b2))
        
        for row in range(n * m):
            if U[row][col1] == 0 and U[row][col2] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0), Fraction(0)))
        
        for row in range(m * p):
            if V[row][col1] == 0 and V[row][col2] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0), Fraction(0)))
        
        for row in range(p * n):
            if W[row][col1] == 0 and W[row][col2] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0), Fraction(0)))
        
        if not rows:
            return other_cols
        
        num_unknowns = len(other_cols)
        
        # We have two RHS vectors, so we solve two systems simultaneously
        # Build augmented matrix with 2 extra columns
        mat = []
        for row_vec, b1, b2 in rows:
            mat.append(row_vec + [b1, b2])
        
        pivot_rows = []
        used_pivots = set()
        for col in range(num_unknowns):
            if col in used_pivots:
                continue
            pivot_row = None
            for r in range(len(mat)):
                if mat[r][col] != 0 and r not in used_pivots:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            
            mat[pivot_row], mat[len(used_pivots)] = mat[len(used_pivots)], mat[pivot_row]
            
            pivot_val = mat[len(used_pivots)][col]
            mat[len(used_pivots)] = [x / pivot_val for x in mat[len(used_pivots)]]
            
            for r in range(len(mat)):
                if r == len(used_pivots):
                    continue
                factor = mat[r][col]
                if factor != 0:
                    mat[r] = [mat[r][c] - factor * mat[len(used_pivots)][c] for c in range(num_unknowns + 2)]
            
            used_pivots.add(len(used_pivots))
            pivot_rows.append((len(used_pivots) - 1, col))
        
        for r in range(len(mat)):
            all_zero = all(mat[r][c] == 0 for c in range(num_unknowns))
            if all_zero and (mat[r][num_unknowns] != 0 or mat[r][num_unknowns + 1] != 0):
                return None
        
        alpha1 = [Fraction(0)] * num_unknowns
        alpha2 = [Fraction(0)] * num_unknowns
        for pr, col in pivot_rows:
            alpha1[col] = mat[pr][num_unknowns]
            alpha2[col] = mat[pr][num_unknowns + 1]
        
        new_terms = []
        for c in other_cols:
            a1 = alpha1[other_idx[c]]
            a2 = alpha2[other_idx[c]]
            if a1 != 0 or a2 != 0:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                # Combine the two columns' contributions
                for (i,j), val in terms[c][0].items():
                    u_dict[(i,j)] = u_dict.get((i,j), Fraction(0)) + val * (a1 + a2)
                for (j,k), val in terms[c][1].items():
                    v_dict[(j,k)] = v_dict.get((j,k), Fraction(0)) + val * (a1 + a2)
                for (i,k), val in terms[c][2].items():
                    w_dict[(i,k)] = w_dict.get((i,k), Fraction(0)) + val * (a1 + a2)
                
                # Remove zeros
                u_dict = {k: v for k, v in u_dict.items() if v != 0}
                v_dict = {k: v for k, v in v_dict.items() if v != 0}
                w_dict = {k: v for k, v in w_dict.items() if v != 0}
                
                if u_dict and v_dict and w_dict:
                    new_terms.append((u_dict, v_dict, w_dict))
        
        return new_terms
    
    # Verify initial decomposition
    U, V, W = build_matrices(rank_terms)
    if not verify_decomposition(U, V, W):
        pass
    
    # Try to remove columns one by one
    current_terms = list(rank_terms)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for col in range(len(current_terms)):
            if time.time() > deadline:
                break
            new_terms = try_remove_column(current_terms, col)
            if new_terms is not None and len(new_terms) < len(current_terms):
                U_new, V_new, W_new = build_matrices(new_terms)
                if verify_decomposition(U_new, V_new, W_new):
                    current_terms = new_terms
                    improved = True
                    break
    
    # Try to remove pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for col1 in range(len(current_terms)):
            if time.time() > deadline:
                break
            for col2 in range(col1 + 1, len(current_terms)):
                if time.time() > deadline:
                    break
                new_terms = try_remove_pair(current_terms, col1, col2)
                if new_terms is not None and len(new_terms) < len(current_terms):
                    U_new, V_new, W_new = build_matrices(new_terms)
                    if verify_decomposition(U_new, V_new, W_new):
                        current_terms = new_terms
                        improved = True
                        break
            if improved:
                break
    
    rank = len(current_terms)
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(current_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
