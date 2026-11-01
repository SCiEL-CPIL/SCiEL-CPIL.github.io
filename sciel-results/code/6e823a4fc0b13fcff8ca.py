import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before the 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j*p+k, W row: k*n+i

    # Build the multiplication tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    # First, try to find a rank-13 solution using a randomized algebraic elimination strategy
    # If that fails, fall back to the known rank-14 Strassen-based solution

    # Strategy: Introduce 13 arbitrary rank-1 terms with symbolic coefficients
    # and solve the resulting linear system for the output factors
    solution = _search_rank_algebraic(n, m, p, T_flat, 13, deadline)
    if solution is not None:
        U, V, W = solution
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Also try rank 12 with the same algebraic approach
    if time.time() < deadline - 20:
        solution = _search_rank_algebraic(n, m, p, T_flat, 12, deadline)
        if solution is not None:
            U, V, W = solution
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank 14
    U14, V14, W14 = _build_rank14_solution()
    if _verify_solution(U14, V14, W14, n, m, p):
        return {"U": U14, "V": V14, "W": W14}

    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _search_rank_algebraic(n, m, p, T_flat, rank, deadline):
    """
    Search for a rank decomposition using algebraic elimination.
    
    Strategy:
    1. Start with random U and V matrices of shape (n*m, rank) and (m*p, rank)
    2. For each column r, we have U[:,r] and V[:,r] fixed
    3. The constraint is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    4. This is linear in W, so we can solve for W given U and V
    5. Then verify if the solution is exact and has nice (integer/half-integer) entries
    """
    max_restarts = 200 if rank <= 13 else 100

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        np.random.seed(restart * 7 + 42)

        # Random initialization of U and V
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)

        # Scale to avoid numerical issues
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)

        # Build the linear system for W
        # For each (a,b), we have: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all c
        # This is a linear system in W[c,r] for each c
        # Actually, it's: for each (a,b,c): sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
        # Let M[a,b,c,r] = U[a,r] * V[b,r], then we need: M[:,:,c,:] @ W[c,:] = T[a,b,c]
        # Wait, that's not quite right. Let me reformulate.
        
        # The constraint is: for all a,b,c:
        #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # This is linear in the entries of W. Let's write it as a linear system.
        # Unknowns: W[c,r] for c in range(p*n), r in range(rank)
        # Total unknowns: p*n*rank
        
        # Equations: for each (a,b,c), one equation
        # Total equations: n*m * m*p * p*n
        
        # Let's build the matrix A such that A @ W_flat = T_flat
        # where W_flat = W.flatten()
        
        num_unknowns = p * n * rank
        num_equations = n * m * m * p * p * n
        
        # Instead of building the full matrix (which could be large), let's use a different approach
        # For each c, the equations involving W[c,:] are:
        #   For each (a,b): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # This is a linear system in W[c,:] for each c
        
        W = np.zeros((p * n, rank))
        
        valid = True
        for c in range(p * n):
            # Build the system for W[c,:]
            # Equations: for each (a,b), sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Let's build a matrix B of shape (n*m*m*p, rank) where
            # B[idx, r] = U[a,r] * V[b,r] for the idx-th (a,b) pair
            # And a vector rhs of shape (n*m*m*p,) where rhs[idx] = T[a,b,c]
            
            B = np.zeros((n * m * m * p, rank))
            rhs = np.zeros(n * m * m * p)
            
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    B[idx, :] = U[a, :] * V[b, :]
                    rhs[idx] = T[a, b, c]
                    idx += 1
            
            # Solve B @ W[c,:] = rhs using least squares
            try:
                W[c, :], residuals, rank_s, sv = np.linalg.lstsq(B, rhs, rcond=None)
            except Exception:
                valid = False
                break
            
            # Check if the solution is exact (within tolerance)
            if not np.allclose(B @ W[c, :], rhs, atol=1e-10):
                valid = False
                break
        
        if not valid:
            continue
        
        # Now verify the full solution
        if _verify_solution(U, V, W, n, m, p):
            # Try to snap to nearest 0.5
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            else:
                return U, V, W

    return None


def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7

    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]

    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]

    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]

    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]

    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]

    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]

    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]

    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7

    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]

    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]

    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]

    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]

    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]

    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]

    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]

    return U.tolist(), V.tolist(), W.tolist()


def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True
