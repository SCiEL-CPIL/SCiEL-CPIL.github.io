"""Rewritten program for matrix multiplication 2x3 by 3x4 using group action / character method."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses the group character method with Z_2^d groups to find a decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Precompute valid triples (u, v, w)
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_triples.append((u, v, w))
                
    valid_set = set(valid_triples)
    
    # Try powers of 2 for the group size (rank)
    # d is the dimension of Z_2^d
    for d in range(1, 10):
        rank = 2 ** d
        
        # Solve for g, h, k in Z_2^d such that g(u) + h(v) + k(w) = 0 for all valid (u,v,w)
        # And g(u) + h(v) + k(w) != 0 for all invalid (u,v,w)
        
        # We first find the null space of the linear system defined by valid triples.
        # Then we check if any solution satisfies the inequality for invalid triples.
        
        # Build the linear system A x = 0 over GF(2)
        # Variables:
        # g_ij for i in n, j in m (n*m variables, each d bits)
        # h_jk for j in m, k in p (m*p variables, each d bits)
        # k_ki for k in p, i in n (p*n variables, each d bits)
        
        num_g = n * m
        num_h = m * p
        num_k = p * n
        total_vars = (num_g + num_h + num_k) * d
        
        # Equations: one for each bit of each valid triple
        # Total equations: len(valid_triples) * d
        num_eqs = len(valid_triples) * d
        
        # We can solve this using Gaussian elimination over GF(2).
        # Since d is small, we can iterate over possible solutions if the null space is small,
        # or just pick a random solution and check.
        
        # Let's use a simple approach:
        # 1. Find a basis for the null space.
        # 2. If the null space is large, sample random solutions.
        # 3. If the null space is small, enumerate them.
        
        # For efficiency, let's just try to find ONE solution that works.
        # We can formulate it as a SAT-like problem or just use the structure.
        
        # Actually, let's just use the fact that if we set g=h=k=0, it works for valid but fails for invalid.
        # We need to perturb it.
        
        # Let's try to solve it properly.
        # We'll build the matrix A of size num_eqs x total_vars.
        
        # To avoid heavy computation, we can process bit by bit.
        # Since the system is identical for each bit component (the structure of indices is the same),
        # we can solve for one bit and then assign different bits to different components?
        # No, the variables are coupled across bits only if we consider the vector sum.
        # But the equation is g + h + k = 0 in Z_2^d. This is d independent equations in Z_2.
        # So we can solve for each bit component independently!
        
        # So we need to find d independent solutions (or just one solution that we can reuse? No, we need d components).
        # Actually, we need g, h, k to be vectors in Z_2^d.
        # Let g = (g_1, ..., g_d), etc.
        # The condition g + h + k = 0 means g_b + h_b + k_b = 0 for each bit b.
        # So we can find d independent solutions (g_b, h_b, k_b) to the scalar equation.
        # Then we combine them.
        # However, we also need the inequality: g + h + k != 0.
        # This means for each invalid triple, the vector sum must be non-zero.
        # This requires that the d scalar sums are not ALL zero.
        
        # Strategy:
        # 1. Find the null space of the scalar system (one bit).
        # 2. Generate d linearly independent vectors in this null space?
        #    Or just pick d random vectors from the null space and check if they form a "separating" set.
        
        # Let's find the null space basis for the scalar case.
        basis = _get_null_space_basis(n, m, p, valid_triples)
        
        if not basis:
            continue
            
        # The null space has dimension dim_null.
        # We need to pick d vectors v_1, ..., v_d from the null space such that
        # for any invalid triple, the sum of the corresponding scalar results is not all zero.
        # Let S_invalid be the set of invalid triples.
        # For each invalid triple t, let f_t(x) be the scalar sum g(u)+h(v)+k(w) for solution x.
        # We need that for each t, the vector (f_t(v_1), ..., f_t(v_d)) is not (0, ..., 0).
        
        # This is possible if the number of invalid triples is less than 2^d?
        # Or if the functions f_t are linearly independent?
        
        # Let's try d=3, 4, etc.
        # We will pick d random vectors from the null space and check.
        
        found = False
        for _ in range(100): # Try 100 random combinations
            # Pick d random vectors from the null space
            # Represent null space vectors as linear combinations of basis vectors
            # Basis vectors are arrays of length total_vars_scalar (which is num_g+num_h+num_k)
            
            # Actually, we can just pick d random solutions to the linear system.
            # A random solution can be generated by picking random values for free variables.
            
            # Let's generate d random solutions.
            sols = []
            for _ in range(d):
                sol = _random_solution(n, m, p, valid_triples, basis)
                sols.append(sol)
                
            # Check if these d solutions separate all invalid triples.
            # For each invalid triple, compute the d scalar sums.
            # If any triple has all 0 sums, it's a fail.
            
            valid = True
            for u in range(n * m):
                for v in range(m * p):
                    for w in range(p * n):
                        if (u, v, w) in valid_set:
                            continue
                            
                        # Compute sum for each solution
                        sums = []
                        for sol in sols:
                            # sol is a flat array of bits for g, h, k
                            # Extract g_u, h_v, k_w
                            # Indices in sol:
                            # g: 0 .. num_g-1
                            # h: num_g .. num_g+num_h-1
                            # k: num_g+num_h .. total-1
                            
                            gu = sol[u]
                            hv = sol[num_g + v]
                            kw = sol[num_g + num_h + w]
                            s = (gu + hv + kw) % 2
                            sums.append(s)
                            
                        if all(s == 0 for s in sums):
                            valid = False
                            break
                if not valid:
                    break
                    
            if valid:
                # Construct matrices
                return _build_matrices_from_sols(sols, n, m, p, d)
                
        # If not found with random, try more or next d
    return _schoolbook(n, m, p)

def _get_null_space_basis(n, m, p, valid_triples):
    """
    Get a basis for the null space of the system g_ij + h_jk + k_ki = 0 over GF(2).
    """
    num_g = n * m
    num_h = m * p
    num_k = p * n
    total_vars = num_g + num_h + num_k
    
    # Build the matrix A (list of rows, each row is a set of variable indices)
    # We use a list of lists of indices for sparse representation
    rows = []
    for (u, v, w) in valid_triples:
        # Equation: g_u + h_v + k_w = 0
        # Indices: u, num_g+v, num_g+num_h+w
        row = [u, num_g + v, num_g + num_h + w]
        rows.append(row)
        
    # Gaussian Elimination over GF(2)
    # We'll represent the matrix as a list of bitmasks or just lists of 0/1.
    # Given the small size, we can use numpy or manual.
    # Let's use a simple list of lists of ints.
    
    # Actually, let's just return the basis of the null space of the transpose?
    # No, we want the null space of A.
    
    # We can use sympy if available, but let's stick to stdlib.
    # We'll implement a simple GF(2) elimination.
    
    # Matrix M of size len(rows) x total_vars
    M = [[0] * total_vars for _ in range(len(rows))]
    for i, row in enumerate(rows):
        for idx in row:
            M[i][idx] = 1
            
    # Perform elimination
    rank = 0
    pivot_cols = []
    for col in range(total_vars):
        # Find pivot
        pivot_row = -1
        for r in range(rank, len(M)):
            if M[r][col] == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
            
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivot_cols.append(col)
        
        # Eliminate
        for r in range(len(M)):
            if r != rank and M[r][col] == 1:
                for c in range(total_vars):
                    M[r][c] ^= M[rank][c]
        rank += 1
        
    # Now M has rank 'rank'. The free variables are those not in pivot_cols.
    # We can construct basis vectors for the null space.
    # Set free variables to 0 except one which is 1, then solve for pivots.
    
    free_cols = [c for c in range(total_vars) if c not in pivot_cols]
    basis = []
    
    for free_col in free_cols:
        vec = [0] * total_vars
        vec[free_col] = 1
        
        # Back substitute
        # M is in reduced row echelon form (RREF)
        for r in range(rank - 1, -1, -1):
            pivot_col = pivot_cols[r]
            # M[r][pivot_col] is 1
            # M[r][free_col] is the coefficient
            if M[r][free_col] == 1:
                vec[pivot_col] = 1 # Since 1 * x_pivot + ... = 0 => x_pivot = sum of others
            
        basis.append(vec)
        
    return basis

def _random_solution(n, m, p, valid_triples, basis):
    """
    Generate a random solution in the null space.
    """
    num_g = n * m
    num_h = m * p
    num_k = p * n
    total_vars = num_g + num_h + num_k
    
    if not basis:
        return [0] * total_vars
        
    # Random linear combination of basis vectors
    coeffs = [1 if np.random.rand() > 0.5 else 0 for _ in range(len(basis))]
    sol = [0] * total_vars
    for i, b in enumerate(basis):
        if coeffs[i]:
            for j in range(total_vars):
                sol[j] ^= b[j]
                
    return sol

def _build_matrices_from_sols(sols, n, m, p, d):
    """
    Build U, V, W from the d solutions.
    """
    num_g = n * m
    num_h = m * p
    num_k = p * n
    rank = 2 ** d
    
    U = [[0.0] * rank for _ in range(num_g)]
    V = [[0.0] * rank for _ in range(num_h)]
    W = [[0.0] * rank for _ in range(num_k)]
    
    # For each c in 0..rank-1, c is a d-bit vector.
    # U[u, c] = 0.5 * (-1)^(dot(c, g_u))
    # g_u is the vector of bits from the d solutions at index u.
    
    # Precompute g, h, k vectors for each index
    g_vecs = [[0] * d for _ in range(num_g)]
    h_vecs = [[0] * d for _ in range(num_h)]
    k_vecs = [[0] * d for _ in range(num_k)]
    
    for bit_idx in range(d):
        sol = sols[bit_idx]
        for u in range(num_g):
            g_vecs[u][bit_idx] = sol[u]
        for v in range(num_h):
            h_vecs[v][bit_idx] = sol[num_g + v]
        for w in range(num_k):
            k_vecs[w][bit_idx] = sol[num_g + num_h + w]
            
    for c in range(rank):
        # Convert c to d-bit vector
        c_bits = [(c >> i) & 1 for i in range(d)]
        
        for u in range(num_g):
            dot = sum((c_bits[i] * g_vecs[u][i]) % 2 for i in range(d))
            U[u][c] = 0.5 * (-1) ** dot
            
        for v in range(num_h):
            dot = sum((c_bits[i] * h_vecs[v][i]) % 2 for i in range(d))
            V[v][c] = 0.5 * (-1) ** dot
            
        for w in range(num_k):
            dot = sum((c_bits[i] * k_vecs[w][i]) % 2 for i in range(d))
            W[w][c] = 0.5 * (-1) ** dot
            
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-END
