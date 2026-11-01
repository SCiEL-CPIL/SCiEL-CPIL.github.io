import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    # Build the known rank 39 solution as fallback
    rank_39 = _build_rank_39()
    
    # Try to find a rank 38 decomposition
    result = _try_rank_38(deadline - start_time)
    
    if result is not None and _verify_decomposition(result, n, m, p):
        return result
    
    # Fallback to rank 39
    return rank_39


def _build_rank_39():
    """Build the known rank 39 decomposition: 5 Strassen blocks (7 each) + 1 schoolbook column (4)."""
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 39)
    V = [[0] * rank for _ in range(m * p)]  # (22, 39)
    W = [[0] * rank for _ in range(p * n)]  # (22, 39)
    
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
    U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
    U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
    U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
    
    return {"U": U, "V": V, "W": W}


def _try_rank_38(remaining_time):
    """
    Try to find a rank 38 decomposition.
    
    Strategy: Use 4 Strassen blocks (28 rank) + a rank 10 decomposition for [2,2,3].
    """
    if remaining_time < 5:
        return None
    
    # Try to find rank 10 for [2,2,3]
    rank_10_sol = _find_rank_10_for_223(remaining_time - 2)
    
    if rank_10_sol is None:
        return None
    
    U_r10, V_r10, W_r10 = rank_10_sol
    
    # Build full rank 38
    rank = 38
    n, m, p = 2, 2, 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # First 4 blocks: Strassen (rank 7 each)
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Last 3 columns (8,9,10): use rank 10 solution
    # V indices for columns 8,9,10: rows 0-1 -> V[8..13], rows 11-12 -> V[19..21]
    # Actually: V[j*p+k] where j in {0,1}, k in {8,9,10}
    # So V[0*11+8]=V[8], V[0*11+9]=V[9], V[0*11+10]=V[10]
    # V[1*11+8]=V[19], V[1*11+9]=V[20], V[1*11+10]=V[21]
    
    # W indices: W[k*n+i] where k in {8,9,10}, i in {0,1}
    # W[8*2+0]=W[16], W[8*2+1]=W[17]
    # W[9*2+0]=W[18], W[9*2+1]=W[19]
    # W[10*2+0]=W[20], W[10*2+1]=W[21]
    
    for i in range(4):
        for r_idx in range(10):
            U[i][r + r_idx] += U_r10[i][r_idx]
    
    # V for the [2,2,3] block: V_local[j*3+k] maps to global V[j*11 + (8+k)]
    for j in range(2):
        for k in range(3):
            for r_idx in range(10):
                V[j * 11 + 8 + k][r + r_idx] += V_r10[j * 3 + k][r_idx]
    
    # W for the [2,2,3] block: W_local[k*2+i] maps to global W[(8+k)*2 + i]
    for k in range(3):
        for i in range(2):
            for r_idx in range(10):
                W[(8 + k) * 2 + i][r + r_idx] += W_r10[k * 2 + i][r_idx]
    
    return {"U": U, "V": V, "W": W}


def _find_rank_10_for_223(time_budget):
    """
    Find a rank 10 decomposition for [2,2,3] (2x2 * 2x3).
    
    U: (4, 10), V: (6, 10), W: (6, 10)
    Tensor: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    
    We need: sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k
    """
    start = time.time()
    deadline = start + time_budget
    
    # Known rank 11 solution: Strassen for first 2 cols (rank 7) + schoolbook for 3rd col (rank 4)
    # Try to find rank 10 by searching over structured solutions
    
    # Approach: Assume the solution has a certain symmetry/structure
    # Let's try a parametric family
    
    # The [2,2,3] tensor can be seen as three copies of [2,2,1] coupled together
    # [2,2,1] has rank 4 with the schoolbook solution
    
    # Let's try to find a rank 10 solution by assuming entries are in {-1, 0, 1, 0.5, -0.5}
    # and using a local search / hill climbing approach
    
    # Initialize with the rank 11 solution (padded to rank 10 by combining two terms)
    # Actually, let's start from scratch with a structured guess
    
    # Try a specific structured solution
    # Based on the pattern that might work for [2,2,3]:
    
    # Let me try a different approach: use the fact that we can write
    # C[i,k] = sum_j A[i,j] B[j,k]
    # and try to find 10 bilinear forms that span this
    
    # Parametric approach: assume U, V, W have entries that are linear combinations
    # of a small set of parameters
    
    # Let's try a direct search with a limited parameter space
    # Assume entries are in {-1, -0.5, 0, 0.5, 1}
    
    # This is a huge search space, so let's use a more clever approach
    
    # Idea: Start from the rank 11 solution and try to merge two rank-1 terms
    # to get rank 10
    
    # Rank 11 solution for [2,2,3]:
    # First 2 columns: Strassen (rank 7)
    # 3rd column: schoolbook (rank 4)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Strassen for columns 0,1
    # m1 = (a+d)(e0+h0)
    U11[0,0] = 1; U11[3,0] = 1
    V11[0,0] = 1; V11[4,0] = 1
    W11[0,0] = 1; W11[3,0] = 1
    
    # m2 = (c+d)e0
    U11[2,1] = 1; U11[3,1] = 1
    V11[0,1] = 1
    W11[1,1] = 1; W11[3,1] = -1
    
    # m3 = a(f0-h0)
    U11[0,2] = 1
    V11[1,2] = 1; V11[4,2] = -1
    W11[2,2] = 1; W11[3,2] = 1
    
    # m4 = d(g0-e0)
    U11[3,3] = 1
    V11[3,3] = 1; V11[0,3] = -1
    W11[0,3] = 1; W11[1,3] = 1
    
    # m5 = (a+b)h0
    U11[0,4] = 1; U11[1,4] = 1
    V11[4,4] = 1
    W11[0,4] = -1; W11[2,4] = 1
    
    # m6 = (c-a)(e0+f0)
    U11[2,5] = 1; U11[0,5] = -1
    V11[0,5] = 1; V11[1,5] = 1
    W11[3,5] = 1
    
    # m7 = (b-d)(g0+h0)
    U11[1,6] = 1; U11[3,6] = -1
    V11[3,6] = 1; V11[4,6] = 1
    W11[0,6] = 1
    
    # Schoolbook for column 2
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # V[2] = B[0,2], V[5] = B[1,2]
    # W[4] = C[0,2], W[5] = C[1,2]
    
    U11[0,7] = 1; V11[2,7] = 1; W11[4,7] = 1
    U11[1,8] = 1; V11[5,8] = 1; W11[4,8] = 1
    U11[2,9] = 1; V11[2,9] = 1; W11[5,9] = 1
    U11[3,10] = 1; V11[5,10] = 1; W11[5,10] = 1
    
    # Now try to find a rank 10 solution by perturbing this
    # and using local search
    
    # Start with a copy and try to merge terms
    U = U11[:, :10].copy()  # Just take first 10 terms as starting point (incomplete)
    V = V11[:, :10].copy()
    W = W11[:, :10].copy()
    
    # This won't work directly. Let's try a different initialization.
    # Use all 11 terms but with one term being a linear combination
    
    # Better: start with the 11-term solution and try to find a linear dependency
    # that allows us to reduce to 10 terms
    
    # Or: use a local search starting from a random rank-10 initialization
    
    # Let's try a simulated annealing / local search approach
    
    # Define the error function
    def compute_error(U, V, W):
        err = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                    err += (val - 1) ** 2
        return err
    
    # Initialize with a structured guess
    # Try to use the Strassen pattern for all 3 columns with some modification
    
    # Let's try a parametric family where we assume certain symmetries
    
    # Actually, let's just do a simple local search
    # Start from the rank 11 solution truncated to 10 (drop the last term)
    # and see if we can adjust to get a valid solution
    
    U = U11[:, :10].astype(float).copy()
    V = V11[:, :10].astype(float).copy()
    W = W11[:, :10].astype(float).copy()
    
    current_err = compute_error(U, V, W)
    
    # Local search: perturb one entry at a time
    step_size = 0.5
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = current_err
    
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick a random matrix and entry
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            i, j = np.random.randint(0, 4), np.random.randint(0, 10)
            old_val = U[i, j]
            new_val = old_val + np.random.choice([-step_size, step_size])
            U[i, j] = new_val
            new_err = compute_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[i, j] = old_val
        elif mat_choice == 1:
            i, j = np.random.randint(0, 6), np.random.randint(0, 10)
            old_val = V[i, j]
            new_val = old_val + np.random.choice([-step_size, step_size])
            V[i, j] = new_val
            new_err = compute_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[i, j] = old_val
        else:
            i, j = np.random.randint(0, 6), np.random.randint(0, 10)
            old_val = W[i, j]
            new_val = old_val + np.random.choice([-step_size, step_size])
            W[i, j] = new_val
            new_err = compute_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[i, j] = old_val
        
        # Check if we found a solution
        if best_err < 1e-10:
            # Round to nearest half-integer
            best_U = np.round(best_U * 2) / 2
            best_V = np.round(best_V * 2) / 2
            best_W = np.round(best_W * 2) / 2
            # Verify
            final_err = compute_error(best_U, best_V, best_W)
            if final_err < 1e-9:
                return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    # Check if best solution is valid
    if best_err < 1e-9:
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        final_err = compute_error(best_U, best_V, best_W)
        if final_err < 1e-9:
            return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    return None


def _verify_decomposition(decomp, n, m, p):
    """Verify that the decomposition correctly decomposes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Check: for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    return True
