"""
Exact rank-minimal tensor decomposition search for 2x5 * 5x6 matrix multiplication.

Strategy:
- The tensor M(n,m,p) has entries T[i,j,k] = e_j (for all i,k) under the convention
  C_ik = sum_j A_ij B_jk.
- A rank-r decomposition is U (nm x r), V (mp x r), W (pn x r) with
  sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1, j2).
- We search for the smallest r >= max(n*m, m*p, p*n) such that an exact
  half-integer solution exists.

Approach:
1. Construct the schoolbook (rank = n*m*p) as a valid baseline.
2. For each candidate rank r from the lower bound upward, attempt to find an exact
   half-integer solution via:
     a. Randomly generate sparse half-integer patterns for W (since p*n is small).
     b. For each W pattern, solve for U and V using linear least squares 
        (the problem is linear in U and V given W).
     c. Snap U and V to nearest half-integers and verify exactness.
     d. If exact, return immediately (since we search in increasing r, first hit is minimal).
3. Use a tight deadline guard to avoid timeout.

Key structural choices:
- Exploit the linearity in U and V given W: this reduces the nonlinear problem
  to a linear one for each W pattern.
- Use random sparse patterns for W to break symmetry and explore the space efficiently.
- Use integer-scaled verification (multiply by 2) to check half-integer exactness.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin: evaluator kills at 150s, keep 35s buffer


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Schoolbook rank
    rank_schoolbook = n * m * p

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

    def check_exact(U, V, W, rank):
        """
        Check if U, V, W are half-integers and satisfy the identity exactly.
        """
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2

        # Check if they are close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False

        # Convert to exact integers
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)

        # Verify identity with integers:
        # sum_r (U_int/2)(V_int/2)(W_int/2) = delta => sum_r U_int V_int W_int = 4 * delta
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i * m:(i + 1) * m, :]
                V_slice = V_int[k * p:(k + 1) * p, :]
                w_val = W_int[k * n + i]

                mat = (U_slice * w_val[None, :]) @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True

    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    min_rank = max(n * m, m * p, p * n)

    # Try ranks from min_rank up to best_rank - 1
    for current_rank in range(min_rank, best_rank):
        if time_left() < 5.0:
            break

        rank_start_time = time.time()
        max_restarts = 200

        for restart in range(max_restarts):
            if time_left() < 5.0:
                break
            if time.time() - rank_start_time > 8.0:
                break

            # Generate random sparse half-integer pattern for W
            # W has shape (p*n, current_rank) = (12, current_rank)
            # Use sparse pattern with entries in {-1, 0, 1} (half-integers scaled by 2)
            W = np.zeros((p * n, current_rank))
            
            # Randomly place non-zero entries
            num_nonzero = random.randint(current_rank // 2, current_rank * 2)
            indices = random.sample(range(p * n * current_rank), min(num_nonzero, p * n * current_rank))
            for idx in indices:
                row = idx // current_rank
                col = idx % current_rank
                W[row, col] = random.choice([-1, 1])
            
            # Ensure at least one non-zero in each row to avoid trivial solutions
            for row in range(p * n):
                if np.all(W[row, :] == 0):
                    W[row, random.randint(0, current_rank - 1)] = random.choice([-1, 1])
            
            # Scale to half-integers
            W = W / 2.0

            # Solve for U and V given W
            # The equation is: sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1, j2)
            # This is linear in U and V. We can fix V and solve for U, or vice versa.
            # Let's fix V to be random half-integers and solve for U.
            
            # Initialize V with random half-integers
            V = np.random.randint(-2, 3, size=(m * p, current_rank)) / 2.0
            
            # For each (i, k), we have:
            # sum_r U[i*m+j1, r] * (V[j2*p+k, r] * W[k*n+i, r]) = delta(j1, j2)
            # Let A_ik = V[k*p:(k+1)*p, :] * W[k*n+i, None, :]  (m x rank)
            # Then U[i*m:(i+1)*m, :] @ A_ik.T = I_m
            
            # This is a linear system for each i. Let's solve it.
            U = np.zeros((n * m, current_rank))
            
            solvable = True
            for i in range(n):
                for k in range(p):
                    # A_ik is (m, rank)
                    A_ik = V[k * p:(k + 1) * p, :] * W[k * n + i, None, :]
                    
                    # We need U[i*m:(i+1)*m, :] @ A_ik.T = I_m
                    # This is: U_slice @ A_ik.T = I_m
                    # So U_slice = I_m @ (A_ik.T)^+ (pseudo-inverse)
                    try:
                        U_slice = np.linalg.pinv(A_ik.T)
                        U[i * m:(i + 1) * m, :] = U_slice
                    except np.linalg.LinAlgError:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if not solvable:
                continue

            # Snap U and V to nearest half-integers
            U_snap = np.round(U * 2) / 2.0
            V_snap = np.round(V * 2) / 2.0
            W_snap = np.round(W * 2) / 2.0

            if check_exact(U_snap, V_snap, W_snap, current_rank):
                best_rank = current_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                break

        if best_rank == current_rank:
            # Found a valid decomposition at this rank; since we search upward,
            # this is the minimal rank we found. Return immediately.
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no better rank was found, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
