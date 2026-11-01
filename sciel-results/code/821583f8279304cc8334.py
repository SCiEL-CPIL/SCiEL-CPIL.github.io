"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses Strassen-based block decomposition (rank 42) and then applies tensor slicing
    analysis and randomized search to find lower rank decompositions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def strassen_based_decomposition():
        """
        Build a rank-42 decomposition using Strassen on 2x2x2 blocks.
        """
        rank = 42
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
        ]
        
        col = 0
        for r_block in range(3):
            for sub in range(2):
                a_col_offset = sub * 2
                b_row_offset = sub * 2
                
                for s_idx in range(7):
                    term = strassen_terms[s_idx]
                    
                    for local_u, coeff in term['U']:
                        a = local_u // 2
                        l = local_u % 2
                        global_u = a * m + (a_col_offset + l)
                        U[global_u][col] += coeff
                    
                    for local_v, coeff in term['V']:
                        l = local_v // 2
                        c = local_v % 2
                        global_v = (b_row_offset + l) * p + (2 * r_block + c)
                        V[global_v][col] += coeff
                    
                    for local_w, coeff in term['W']:
                        a = local_w // 2
                        c = local_w % 2
                        global_w = (2 * r_block + c) * n + a
                        W[global_w][col] += coeff
                    
                    col += 1
        
        return U, V, W, rank

    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def analyze_slice_space(U, V, W, r):
        """
        Analyze the tensor using slice method. For each index of U, V, or W,
        form the corresponding slice matrix and check its rank. This helps
        identify if the decomposition is minimal for its structure.
        """
        # Convert to numpy for efficient linear algebra
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each fixed index of U (i*m+j), the slice is V[:, t] * W[:, t]^T
        # Actually, the tensor is T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        
        # Let's analyze the structure differently.
        # We can try to find if any columns are linearly dependent in a way
        # that allows rank reduction.
        
        # Check if we can remove terms by checking linear dependencies
        # among the "term tensors"
        
        return True  # Placeholder for now

    def try_eliminate_term(U, V, W, r, t):
        """
        Try to eliminate term t by expressing its tensor contribution as a linear
        combination of the other terms. Solves a linear system in the coefficients.
        Returns (True, c_solution) if successful, (False, None) otherwise.
        """
        other_terms = [s for s in range(r) if s != t]
        num_unknowns = len(other_terms)
        num_equations = n * m * p
        
        A_matrix = np.zeros((num_equations, num_unknowns))
        b_vector = np.zeros(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b_vector[eq_idx] = U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    for u_idx, s in enumerate(other_terms):
                        A_matrix[eq_idx, u_idx] = U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                    eq_idx += 1
        
        try:
            c_solution, residuals, rank, sv = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
            if np.allclose(A_matrix @ c_solution, b_vector, atol=1e-9):
                return True, c_solution
        except:
            pass
        
        return False, None

    def reduce_rank(U, V, W, r):
        """
        Iteratively try to eliminate redundant terms. When term t is eliminated,
        its contribution is redistributed to the remaining terms by scaling their
        U-columns. This reduces the rank by 1 per successful elimination.
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        eliminated = set()
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            for t in range(r):
                if t in eliminated:
                    continue
                if time.time() > deadline:
                    break
                
                can_elim, c_solution = try_eliminate_term(U, V, W, r, t)
                if can_elim:
                    other_terms = [s for s in range(r) if s != t]
                    for u_idx, s in enumerate(other_terms):
                        scale = 1 + c_solution[u_idx]
                        for i in range(len(U)):
                            U[i][s] *= scale
                    
                    eliminated.add(t)
                    changed = True
        
        if not eliminated:
            return U, V, W, r
        
        keep = [i for i in range(r) if i not in eliminated]
        new_r = len(keep)
        new_U = [[U[i][c] for c in keep] for i in range(len(U))]
        new_V = [[V[i][c] for c in keep] for i in range(len(V))]
        new_W = [[W[i][c] for c in keep] for i in range(len(W))]
        
        if verify(new_U, new_V, new_W, new_r):
            return new_U, new_V, new_W, new_r
        
        return U, V, W, r

    def randomized_search(U0, V0, W0, r0, max_iters=1000):
        """
        Randomized search for lower rank decompositions.
        Start from the current decomposition and try random perturbations
        followed by rank reduction.
        """
        best_U, best_V, best_W, best_r = U0, V0, W0, r0
        best_r = r0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed copy
            U = [row[:] for row in U0]
            V = [row[:] for row in V0]
            W = [row[:] for row in W0]
            r = r0
            
            # Apply small random perturbations to some entries
            for _ in range(3):
                # Randomly perturb U, V, or W
                which = random.choice(['U', 'V', 'W'])
                if which == 'U':
                    idx = random.randint(0, len(U)-1)
                    col = random.randint(0, r-1)
                    U[idx][col] += random.uniform(-0.5, 0.5)
                elif which == 'V':
                    idx = random.randint(0, len(V)-1)
                    col = random.randint(0, r-1)
                    V[idx][col] += random.uniform(-0.5, 0.5)
                else:
                    idx = random.randint(0, len(W)-1)
                    col = random.randint(0, r-1)
                    W[idx][col] += random.uniform(-0.5, 0.5)
            
            # Try to reduce rank
            new_U, new_V, new_W, new_r = reduce_rank(U, V, W, r)
            
            if new_r < best_r:
                best_U, best_V, best_W, best_r = new_U, new_V, new_W, new_r
                U0, V0, W0, r0 = best_U, best_V, best_W, best_r
        
        return best_U, best_V, best_W, best_r

    # Start with the Strassen-based decomposition
    U, V, W, r = strassen_based_decomposition()
    
    # Verify the base decomposition
    if not verify(U, V, W, r):
        # Fall back to schoolbook
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Try to reduce rank via linear elimination
    U, V, W, r = reduce_rank(U, V, W, r)
    
    # If we have time, try randomized search for further improvements
    if time.time() < deadline - 20:
        U, V, W, r = randomized_search(U, V, W, r, max_iters=500)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
