import random
import numpy as np
import time
import itertools
from scipy.optimize import linprog

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 140s limit to be safe (150s kill)
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Target ranks to try
    # Lower bound is max(n*m, m*p, p*n) = 6
    # Known rank for 2x2x3 is 10
    target_ranks = [10, 11]
    
    # Candidate values for matrix entries
    # Matrix multiplication tensors often decompose with small integers/half-integers
    # We restrict search to this set to make combinatorial search feasible
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()
    T_flat = T_target.flatten()
    
    def verify_decomposition(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Returns True if exact decomposition
        """
        rank = U.shape[1]
        # T_prime_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # Vectorize:
        # T_prime_flat[a, b, c]
        # Let's compute for each r, add outer product
        T_prime = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            T_prime += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        
        return np.allclose(T_prime, T_target, atol=1e-9)

    def search_rank(target_rank):
        if time.time() > deadline - 5.0:
            return None
            
        # We search for sparse decompositions first.
        # Most known low-rank decompositions for matrix multiplication are sparse.
        # We will try to fill the matrices with 0s and non-zeros from candidates.
        
        # Total variables: (n*m + m*p + p*n) * target_rank
        # For rank 10: (4 + 6 + 6) * 10 = 160 variables.
        # This is too large for brute force over 7 candidates.
        
        # Strategy:
        # 1. Use the fact that rank-1 terms are outer products.
        # 2. Often, we can fix some structure.
        # 3. Alternatively, use a heuristic local search on the discrete domain.
        
        # Let's try a randomized local search on the discrete lattice.
        # Initialize with random sparse matrices.
        
        random.seed(42 + target_rank)
        np.random.seed(42 + target_rank)
        
        def random_sparse_matrix(rows, cols, density, candidates):
            M = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    if random.random() < density:
                        M[i, j] = random.choice(candidates)
            return M

        best_loss = float('inf')
        best_solution = None
        
        # Try a few initializations
        for attempt in range(50):
            if time.time() > deadline - 2.0:
                break
                
            # Start with very sparse matrices
            density = 0.2
            U = random_sparse_matrix(n*m, target_rank, density, candidates)
            V = random_sparse_matrix(m*p, target_rank, density, candidates)
            W = random_sparse_matrix(p*n, target_rank, density, candidates)
            
            # Local search: randomly pick an element, change its value, if loss decreases, keep it.
            # Loss is sum of squared errors.
            
            current_U, current_V, current_W = U, V, W
            current_loss = float('inf')
            
            # Calculate initial loss
            T_prime = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_prime += np.einsum('i,j,k->ijk', current_U[:, r], current_V[:, r], current_W[:, r])
            current_loss = np.sum((T_prime - T_target)**2)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_solution = (current_U.copy(), current_V.copy(), current_W.copy())
                if current_loss < 1e-9:
                    return best_solution
                    
            # Local search iterations
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline - 2.0:
                    break
                    
                # Pick a random matrix and element
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    mat = current_U
                    rows = n*m
                elif mat_choice == 1:
                    mat = current_V
                    rows = m*p
                else:
                    mat = current_W
                    rows = p*n
                    
                i = random.randint(0, rows - 1)
                j = random.randint(0, target_rank - 1)
                
                old_val = mat[i, j]
                new_val = random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                mat[i, j] = new_val
                
                # Recalculate loss
                # Optimization: Only recalculate the term involving the changed rank?
                # No, the changed element is in rank j.
                # T_prime = sum_r ...
                # The term for rank j changes.
                # Let's just recalculate the whole thing for simplicity, it's small (4x6x6=144 elements)
                T_prime = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_prime += np.einsum('i,j,k->ijk', current_U[:, r], current_V[:, r], current_W[:, r])
                new_loss = np.sum((T_prime - T_target)**2)
                
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_solution = (current_U.copy(), current_V.copy(), current_W.copy())
                        if current_loss < 1e-9:
                            return best_solution
                else:
                    # Revert
                    mat[i, j] = old_val
                    
        return best_solution if best_loss < 1e-9 else None

    for rank in target_ranks:
        if time.time() > deadline - 5.0:
            break
        result = search_rank(rank)
        if result is not None:
            U, V, W = result
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            best_rank = rank
            break

    return {"U": best_U, "V": best_V, "W": best_W}
