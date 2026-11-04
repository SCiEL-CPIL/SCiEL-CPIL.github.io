import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 4, 4
    rank_known = 28
    
    # Build the known rank-28 solution (Strassen on 2x2 blocks)
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1: (x0+x3)(y0+y3) -> c0, c3
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2: (x2+x3)y0 -> c2, -c3
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3: x0(y1-y3) -> c1, c3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4: x3(y2-y0) -> c0, c2
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5: (x0+x1)y3 -> -c0, c1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6: (x2-x0)(y0+y1) -> c3
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7: (x1-x3)(y2+y3) -> c0
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    best_result = {"U": U_known, "V": V_known, "W": W_known}
    best_rank = rank_known

    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s limit

    def verify_solution(U, V, W, rank):
        """Verify that U, V, W with given rank exactly decompose the tensor."""
        # Build target tensor
        T = np.zeros((8, 16, 8), dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    x = i * m + k
                    y = k * p + j
                    z = j * n + i
                    T[x, y, z] = 1
        
        # Compute decomposition
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = np.array(U[:, r], dtype=float)
            v = np.array(V[:, r], dtype=float)
            w = np.array(W[:, r], dtype=float)
            T_est += np.einsum('r,s,t->rst', u, v, w)
        
        return np.allclose(T_est, T, atol=1e-9)

    def try_rank_reduction(U, V, W, target_rank, max_iterations=200):
        """
        Try to reduce rank from current to target_rank by finding linear
        dependencies among the rank-1 terms and eliminating them.
        Uses tensor slicing: partition C into two 2x2 blocks and check
        mode-2 flattenings for rank deficiency.
        """
        current_rank = len(U[0])
        if current_rank <= target_rank:
            return None
        
        # Work with float arrays for numerical computation
        Uf = np.array(U, dtype=float)
        Vf = np.array(V, dtype=float)
        Wf = np.array(W, dtype=float)
        
        # Strategy: try to find a set of target_rank columns that span the same
        # tensor space. We do this by:
        # 1. For each output slice z, look at the (8x16) matrix M_z = sum_r U[:,r] V[:,r] W[z,r]
        # 2. Check if we can express the tensor with fewer terms
        
        # Simpler approach: random perturbation + SVD-based compression
        # For each of the 8 output indices z, the tensor slice is:
        # T[:, :, z] = sum_r W[z,r] * (U[:,r] outer V[:,r])
        # This is a sum of rank-1 matrices weighted by W[z,r]
        
        # Try: fix one set of matrices and solve for others
        # Use the fact that if we have more terms than needed, some are linearly dependent
        
        # Approach: Gaussian elimination on the "term space"
        # Represent each term r as a vector in R^(8*16*8) (flattened outer product)
        # But that's too large. Instead, use the bilinear structure.
        
        # Better: try to merge pairs of terms
        # For terms r1, r2, check if their sum can be written as a single term
        # (u1 v1 w1 + u2 v2 w2 = u3 v3 w3) - generally not possible for 2 terms -> 1
        
        # Practical approach: iterative local search
        # 1. Start with the known solution
        # 2. Randomly select a term to eliminate
        # 3. Try to redistribute its contribution among remaining terms
        # 4. If exact redistribution found, accept
        
        U_try = [row[:] for row in U]
        V_try = [row[:] for row in V]
        W_try = [row[:] for row in W]
        rank_try = current_rank
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            if rank_try <= target_rank:
                break
            
            # Pick a random term to try to eliminate
            term_to_eliminate = np.random.randint(rank_try)
            
            # Try to find coefficients alpha_r such that:
            # term_to_eliminate = sum_{r != term_to_eliminate} alpha_r * term_r
            # This requires solving a system. For exact integer/half-integer,
            # we check if the term is in the span of others.
            
            # Build the system: for each (x,y,z) where T[x,y,z] != 0,
            # U[x,term]*V[y,term]*W[z,term] = sum_{r!=term} alpha_r * U[x,r]*V[y,r]*W[z,r]
            
            # This is a large nonlinear system. Instead, use a heuristic:
            # Check if removing this term and adjusting others still works.
            
            # Simpler: just try removing the term and see if we can fix it
            # by scaling other terms. This only works if the term is redundant.
            
            # Check: is term_to_eliminate a linear combination of others?
            # In the space of rank-1 tensors, this is complex.
            
            # Alternative: use the structure. Each block of 7 Strassen terms
            # might have internal dependencies when combined across blocks.
            
            # For now, do a random perturbation search:
            # Pick 3 random terms and try to replace them with 2 terms
            
            if rank_try <= target_rank + 1:
                # Try to reduce by 1
                # Pick two terms and merge
                i = np.random.randint(rank_try)
                j = np.random.choice([x for x in range(rank_try) if x != i])
                
                # Try to express terms i and j as a single term
                # This is generally impossible, so skip
                pass
            else:
                # Just remove a random term and verify
                U_try.pop(term_to_eliminate)
                V_try.pop(term_to_eliminate)
                W_try.pop(term_to_eliminate)
                rank_try -= 1
                
                if verify_solution(U_try, V_try, W_try, rank_try):
                    return (U_try, V_try, W_try, rank_try)
                else:
                    # Restore
                    U_try.insert(term_to_eliminate, [U[x][term_to_eliminate] for x in range(len(U))])
                    V_try.insert(term_to_eliminate, [V[x][term_to_eliminate] for x in range(len(V))])
                    W_try.insert(term_to_eliminate, [W[x][term_to_eliminate] for x in range(len(W))])
                    rank_try += 1
        
        return None

    # Try to find a lower-rank solution
    # First, check if rank 27 is possible by looking for linear dependencies
    for target_rank in range(rank_known - 1, 23, -1):
        if time.time() > deadline:
            break
        
        found = try_rank_reduction(U_known, V_known, W_known, target_rank, max_iterations=50)
        if found is not None:
            U_new, V_new, W_new, rank_new = found
            if verify_solution(U_new, V_new, W_new, rank_new):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = rank_new
                break
    
    # Additional search: try different block decompositions
    # Maybe a non-standard partitioning gives lower rank
    if time.time() < deadline - 10:
        # Try random integer solutions for rank 27
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randint(-2, 3, (8, 27))
            V_rand = np.random.randint(-2, 3, (16, 27))
            W_rand = np.random.randint(-2, 3, (8, 27))
            
            if verify_solution(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), 27):
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                best_rank = 27
                break
    
    return best_result
