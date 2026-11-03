"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def check_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct using exact arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    def compute_M_vectorized(U, V, rank):
        """Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c] using numpy."""
        # U shape: (n*m, rank), V shape: (m*p, rank)
        # For each c, U_c has shape (n*m,), V_c has shape (m*p,)
        # Reshape U_c to (n, m), V_c to (m, p), then M[:, :, c] = U_c_reshaped @ V_c_reshaped
        M = np.zeros((n, p, rank), dtype=int)
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            M[:, :, c] = U_c @ V_c
        return M

    def try_solve_W(M, rank, strategy='greedy'):
        """
        Try to find W such that for each (i,k): sum_c M[i,k,c]*W[k*n+i,c] = 1
        W entries must be half-integers (i.e., in {0, ±1/2, ±1, ±3/2, ...})
        M entries are integers.
        
        Returns W as a list of lists, or None if no solution found.
        """
        W = [[0.0] * rank for _ in range(p * n)]
        
        if strategy == 'greedy':
            # For each (i,k), try to find a single column c where M[i,k,c] is ±1 or ±2
            # and set W[k*n+i][c] = 1/M[i,k,c]
            for i in range(n):
                for k in range(p):
                    found = False
                    # Try columns in order of preference: |M|=1 first, then |M|=2
                    for target_abs in [1, 2]:
                        for c in range(rank):
                            m_val = M[i, k, c]
                            if abs(m_val) == target_abs:
                                W[k * n + i][c] = 1.0 / m_val
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        return None
            return W
        elif strategy == 'multi':
            # Allow up to 2 non-zero W entries per (i,k)
            for i in range(n):
                for k in range(p):
                    # Find all valid columns
                    valid_cols = []
                    for c in range(rank):
                        m_val = M[i, k, c]
                        if m_val in (1, -1, 2, -2):
                            valid_cols.append(c)
                    if not valid_cols:
                        return None
                    # Try single column first
                    for c in valid_cols:
                        m_val = M[i, k, c]
                        W[k * n + i][c] = 1.0 / m_val
                        # Check if this is consistent (it always is by construction)
                        # Just pick the first one
                        break
                    # If we need multiple, try pairs
                    # For simplicity, just use the first valid column
            return W
        return None

    def random_sparse_matrix(rows, rank, p_nonzero):
        """Generate a random sparse matrix with entries in {-1, 0, 1}."""
        M = np.zeros((rows, rank), dtype=int)
        for r in range(rows):
            for c in range(rank):
                if random.random() < p_nonzero:
                    M[r][c] = random.choice([-1, 1])
        return M

    # Strategy 1: Random search with vectorized M computation
    for target_rank in range(24, 36):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        max_attempts = 300 if target_rank <= 26 else 150
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15.0:
                break
            
            # Vary density
            p_nonzero_U = random.choice([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            p_nonzero_V = random.choice([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            
            U = random_sparse_matrix(n * m, target_rank, p_nonzero_U)
            V = random_sparse_matrix(m * p, target_rank, p_nonzero_V)
            
            M = compute_M_vectorized(U, V, target_rank)
            
            W_list = try_solve_W(M, target_rank, strategy='greedy')
            if W_list is None:
                continue
            
            U_list = U.tolist()
            V_list = V.tolist()
            
            if check_decomposition(U_list, V_list, W_list, target_rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: Structured perturbation of schoolbook
    # Try to merge some schoolbook columns
    if time.time() < deadline - 20.0:
        for target_rank in range(24, min(best_rank, 32)):
            if time.time() > deadline - 15.0:
                break
            
            # Start from schoolbook and try to merge columns
            # Schoolbook has 36 columns. We want to reduce to target_rank.
            # Idea: randomly group schoolbook columns and try to combine them.
            
            max_attempts = 100
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 15.0:
                    break
                
                # Create a random partition of the 36 schoolbook indices into target_rank groups
                indices = list(range(rank_schoolbook))
                random.shuffle(indices)
                
                # Assign each index to a group (0 to target_rank-1)
                group_assign = {}
                for idx in indices:
                    # Round-robin assignment with some randomness
                    base_group = idx % target_rank
                    # Sometimes shift to create different structures
                    if random.random() < 0.3:
                        base_group = (base_group + random.randint(1, min(2, target_rank-1))) % target_rank
                    group_assign[idx] = base_group
                
                # Build new U, V, W by summing contributions from grouped schoolbook columns
                U_new = np.zeros((n * m, target_rank), dtype=int)
                V_new = np.zeros((m * p, target_rank), dtype=int)
                W_new = np.zeros((p * n, target_rank), dtype=int)
                
                for col_idx in range(rank_schoolbook):
                    # Extract i, j, k from col_idx
                    # col_idx = i * m * p + j * p + k
                    i = col_idx // (m * p)
                    rem = col_idx % (m * p)
                    j = rem // p
                    k = rem % p
                    
                    g = group_assign[col_idx]
                    
                    U_new[i * m + j][g] += 1
                    V_new[j * p + k][g] += 1
                    W_new[k * n + i][g] += 1
                
                # Now we need to check if this works. The issue is that when we sum
                # multiple schoolbook terms into one column, the product U*V*W may not
                # equal the desired value. We need to scale W appropriately.
                
                # For each column g, let's compute what the "effective" contribution is.
                # For a given (i,j,k), the contribution from column g is:
                #   U_new[i*m+j, g] * V_new[j*p+k, g] * W_new[k*n+i, g]
                # We need sum_g of this to equal 1 for each (i,j,k).
                
                # Let's check and adjust W if needed
                valid = True
                W_adjusted = [[0.0] * target_rank for _ in range(p * n)]
                
                for i in range(n):
                    for k in range(p):
                        # For this (i,k), we need sum_g M[i,k,g] * W[k*n+i,g] = 1
                        # where M[i,k,g] = sum_j U_new[i*m+j,g] * V_new[j*p+k,g]
                        M_ik = np.zeros(target_rank, dtype=int)
                        for j in range(m):
                            M_ik += U_new[i * m + j] * V_new[j * p + k]
                        
                        # Try to solve: sum_g M_ik[g] * W_val[g] = 1
                        # with W_val[g] being half-integer
                        found_solution = False
                        # Try single non-zero
                        for g in range(target_rank):
                            if M_ik[g] in (1, -1, 2, -2):
                                W_adjusted[k * n + i][g] = 1.0 / M_ik[g]
                                found_solution = True
                                break
                        if not found_solution:
                            # Try two non-zeros
                            for g1 in range(target_rank):
                                if M_ik[g1] not in (1, -1, 2, -2):
                                    continue
                                w1 = 1.0 / M_ik[g1]
                                # Check if w1 is half-integer
                                if abs(w1 * 2 - round(w1 * 2)) > 1e-9:
                                    continue
                                for g2 in range(g1 + 1, target_rank):
                                    if M_ik[g2] not in (1, -1, 2, -2):
                                        continue
                                    # We need w1 + M_ik[g2]*w2 = 1, so w2 = (1 - w1)/M_ik[g2]
                                    w2 = (1.0 - w1) / M_ik[g2]
                                    if abs(w2 * 2 - round(w2 * 2)) < 1e-9:
                                        W_adjusted[k * n + i][g1] = w1
                                        W_adjusted[k * n + i][g2] = w2
                                        found_solution = True
                                        break
                                if found_solution:
                                    break
                            if found_solution:
                                continue
                        if not found_solution:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    U_list = U_new.tolist()
                    V_list = V_new.tolist()
                    
                    if check_decomposition(U_list, V_list, W_adjusted, target_rank):
                        best_result = {"U": U_list, "V": V_list, "W": W_adjusted}
                        best_rank = target_rank
                        break
            
            if best_rank == target_rank:
                break
    
    return best_result
# EVOLVE-BLOCK-END
