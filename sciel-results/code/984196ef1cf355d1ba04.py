import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, r):
        """Verify that U, V, W with rank r exactly decompose the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_solve_for_W(U, V, r):
        """Given U and V, solve for W that makes the decomposition exact."""
        num_eqs = n * m * p
        num_unknowns = p * n * r
        
        A_mat = np.zeros((num_eqs, num_unknowns))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    for c in range(r):
                        coeff = U[i * m + j, c] * V[j * p + k, c]
                        if coeff != 0:
                            A_mat[eq_idx, w_row * r + c] = coeff
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return None
        
        W = W_flat.reshape(p * n, r)
        
        # Check if W is close to half-integers
        W_half = W * 2
        W_rounded = np.round(W_half)
        
        error = np.max(np.abs(W_half - W_rounded))
        if error < 1e-4:
            W_candidate = W_rounded / 2.0
            if verify_decomposition(U, V, W_candidate, r):
                return W_candidate
        return None
    
    def generate_structured_U_V(r, method=0):
        """Generate structured U and V matrices that are more likely to yield integer/half-integer W."""
        if method == 0:
            # Block-diagonal style: each rank element handles a subset of (i,j,k)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            # Assign rank elements to cover all (i,j,k) triples
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if idx < r:
                            U[i * m + j, idx] = 1
                            V[j * p + k, idx] = 1
                            idx += 1
            # Add some randomization
            if idx < r:
                for c in range(idx, r):
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, m * p - 1)
                    U[row, c] = random.choice([-1, 1])
                    V[col, c] = random.choice([-1, 1])
            return U, V
        elif method == 1:
            # Strassen-like: use Kronecker products
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            # Try to embed a smaller known decomposition
            # For now, use a pattern based on XOR-like operations
            for c in range(r):
                # Create a pattern based on the rank index
                mask_u = (c // 3) % 2
                mask_v = (c // 2) % 2
                for i in range(n):
                    for j in range(m):
                        if ((i ^ j) & mask_u) == (c % 2):
                            U[i * m + j, c] = 1 if (c % 3) != 0 else -1
                for j in range(m):
                    for k in range(p):
                        if ((j ^ k) & mask_v) == (c % 2):
                            V[j * p + k, c] = 1 if (c % 3) != 0 else -1
            return U, V
        elif method == 2:
            # Random but with more structure: sparse with controlled density
            density = 0.3
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for c in range(r):
                # Each column has a few non-zero entries
                num_nonzero_u = max(1, int(n * m * density))
                num_nonzero_v = max(1, int(m * p * density))
                u_rows = random.sample(range(n * m), min(num_nonzero_u, n * m))
                v_rows = random.sample(range(m * p), min(num_nonzero_v, m * p))
                for row in u_rows:
                    U[row, c] = random.choice([-1, 1])
                for row in v_rows:
                    V[row, c] = random.choice([-1, 1])
            return U, V
        else:
            # Pure random
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            return U, V
    
    # Try to find lower rank decompositions
    # Search from lower ranks first to find the minimum
    for r in range(40, 56):
        if time.time() > deadline - 15:
            break
            
        # Number of attempts per rank
        attempts = 80
        
        for attempt in range(attempts):
            if time.time() > deadline - 8:
                break
            
            method = attempt % 4
            
            U, V = generate_structured_U_V(r, method)
            
            # Try to solve for W
            W = try_solve_for_W(U, V, r)
            
            if W is not None:
                # Found a valid decomposition
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a valid one at rank r, we can return it
        # But we want the minimum, so continue searching higher ranks only if needed
    
    return best_result
# EVOLVE-BLOCK-END
