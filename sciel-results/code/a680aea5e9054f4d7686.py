"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 145  # Safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        if rank == 0:
            return False
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def solve_for_W(U, V, rank):
        """
        Given U and V, solve for W such that sum_r U_ij_r V_jk_r W_ki_r = 1.
        Returns W as a list of lists or None if no integer solution found near the real solution.
        """
        # Number of unknowns in W: p * n * rank
        num_unknowns = pn * rank
        # Number of equations: n * m * p
        num_eqs = n * m * p
        
        # Build the linear system A w = b
        # A has shape (num_eqs, num_unknowns)
        # b has shape (num_eqs,)
        # Equation index e corresponds to (i, j, k)
        # Unknown index u corresponds to (k, i, r) -> W[k*n + i, r]
        
        # We can construct A sparsely or densely. Since dimensions are small, dense is fine.
        A = np.zeros((num_eqs, num_unknowns), dtype=np.int64)
        b = np.ones(num_eqs, dtype=np.int64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    for r in range(rank):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        if u_val != 0 and v_val != 0:
                            # Index in W vector: (k*n + i) * rank + r
                            w_idx = (k * n + i) * rank + r
                            A[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve A w = b using least squares
        # Since the system is under-determined (usually), we want a solution with small norm
        # or close to integers.
        try:
            w_real, residuals, rank_solver, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Check if the solution is close to integers
        # Tolerance for "close to integer"
        tol = 1e-6
        if not np.all(np.abs(w_real - np.round(w_real)) < tol):
            return None
            
        w_int = np.round(w_real).astype(np.int64)
        
        # Verify the integer solution exactly
        # Reshape w_int to W
        W_flat = w_int.reshape(pn, rank)
        
        # Re-verify with the original U, V
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = W_flat
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.all(T == 1):
            # Convert back to list of lists for the required format
            W_list = W_np.tolist()
            return W_list
        else:
            return None

    def try_rank(target_rank, max_attempts, value_range):
        """
        Try to find a decomposition of the given rank by fixing U, V and solving for W.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate random U and V
            U = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            
            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}
                
        return None

    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = 21
    # Start from lower bound up to schoolbook rank - 1
    
    # We prioritize lower ranks.
    # We use a mix of value ranges.
    # Small integers are more likely to yield integer solutions for W.
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Increase attempts for lower ranks as they are harder
        # But also cap it to avoid spending too much time on impossible ranks
        attempts = 200 if target_rank < 25 else 100
        
        # Try with values {-1, 1}
        result = try_rank(target_rank, attempts, (-1, 1))
        if result:
            best = result
            best_rank = target_rank
            break
            
        # Try with values {0, 1}
        result = try_rank(target_rank, attempts // 2, (0, 1))
        if result:
            best = result
            best_rank = target_rank
            break

    return best
# EVOLVE-BLOCK-END
