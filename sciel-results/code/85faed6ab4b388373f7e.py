"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 120  # 120s limit, keep safety margin
    
    # Build support tensor
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    # All possible (a,b,c) tuples
    all_tuples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    support_set = set(support)
    
    # Schoolbook decomposition (rank 20)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for idx, (a, b, c) in enumerate(support):
        U_school[a, idx] = 1.0
        V_school[b, idx] = 1.0
        W_school[c, idx] = 1.0
    
    def to_list(arr):
        return arr.tolist()
    
    # Function to compute error for given U, V, W
    def compute_error(U, V, W):
        max_err = 0.0
        for (a, b, c) in support:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        # Check non-support entries
        for (a, b, c) in all_tuples:
            if (a, b, c) not in support_set:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Try to find lower rank decompositions
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    # Try ranks from 10 to 19
    for r in range(10, rank_school):
        if time.time() > deadline - 30:  # Keep 30s margin
            break
        
        # Try multiple random initializations
        max_attempts = 20 if r >= 15 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: fix V, W
                # For each a, we want: sum_r' U[a,r'] * (V[b,r']*W[c,r']) = T[a,b,c]
                # This is a least squares problem for each a
                for a in range(n*m):
                    # Build constraint matrix
                    # For each (b,c) where there's a constraint involving a
                    rows = []
                    targets = []
                    for (a2, b, c) in support:
                        if a2 == a:
                            row = V[b, :] * W[c, :]
                            rows.append(row)
                            targets.append(1.0)
                    # Also add constraints from non-support to encourage zeros
                    # Sample some non-support constraints
                    for (a2, b, c) in all_tuples:
                        if a2 == a and (a2, b, c) not in support_set:
                            row = V[b, :] * W[c, :]
                            rows.append(row)
                            targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        # Solve least squares
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a, :] = sol
                
                # Update V: fix U, W
                for b in range(m*p):
                    rows = []
                    targets = []
                    for (a, b2, c) in support:
                        if b2 == b:
                            row = U[a, :] * W[c, :]
                            rows.append(row)
                            targets.append(1.0)
                    for (a, b2, c) in all_tuples:
                        if b2 == b and (a, b2, c) not in support_set:
                            row = U[a, :] * W[c, :]
                            rows.append(row)
                            targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b, :] = sol
                
                # Update W: fix U, V
                for c in range(p*n):
                    rows = []
                    targets = []
                    for (a, b, c2) in support:
                        if c2 == c:
                            row = U[a, :] * V[b, :]
                            rows.append(row)
                            targets.append(1.0)
                    for (a, b, c2) in all_tuples:
                        if c2 == c and (a, b, c2) not in support_set:
                            row = U[a, :] * V[b, :]
                            rows.append(row)
                            targets.append(0.0)
                    
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c, :] = sol
                
                # Check convergence
                err = compute_error(U, V, W)
                if err < 1e-8:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
        if best_rank == r:
            break
    
    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
