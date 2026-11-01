import numpy as np
import time
import random
import itertools

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks exploiting the shared A matrix structure.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: output columns k in {0,1}, rank indices 0..6
    # Strassen for 2x2 * 2x2: A is 2x2, B1 is 2x2 (cols 0,1 of B)
    # U rows: [A00, A01, A10, A11] (indices 0,1,2,3)
    # V rows for block 1: [B00, B01, B10, B11] (indices 0,1,2,3)
    # W rows for block 1: [C00, C01, C10, C11] (indices 0,1,2,3)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)(B10) -> C10, C11
    U[2][1] = 1; U[3][1] = 1
    V[2][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = (A00)(B10-B11) -> C00, C01
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = (A11)(B01-B00) -> C00, C10
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)(B11) -> C01, C11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C10, C00
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = -1
    
    # M7 = (A01-A11)(B10+B11) -> C01, C10
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[2][6] = -1
    
    # Block 2: output columns k in {2,3}, rank indices 7..13
    # V rows for block 2: [B02, B03, B12, B13] (indices 4,5,6,7)
    # W rows for block 2: [C02, C03, C12, C13] (indices 4,5,6,7)
    
    # M8 = (A00+A11)(B02+B13) -> C02, C13
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M9 = (A10+A11)(B12) -> C12, C13
    U[2][8] = 1; U[3][8] = 1
    V[6][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    # M10 = (A00)(B12-B13) -> C02, C03
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[4][9] = 1; W[5][9] = 1
    
    # M11 = (A11)(B03-B02) -> C02, C12
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[6][10] = 1
    
    # M12 = (A00+A01)(B13) -> C03, C13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[5][11] = 1; W[7][11] = -1
    
    # M13 = (A10-A00)(B02+B03) -> C12, C02
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[6][12] = 1; W[4][12] = -1
    
    # M14 = (A01-A11)(B12+B13) -> C03, C12
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[5][13] = 1; W[6][13] = -1
    
    return {"U": U, "V": V, "W": W}

def _search_rank_12():
    """
    Structured search for a rank-12 decomposition.
    We assume the decomposition can be formed by 12 rank-1 terms.
    Each term r has:
    - u_r: vector in R^4 (coeffs for A00, A01, A10, A11)
    - v_r: vector in R^8 (coeffs for B00..B13)
    - w_r: vector in R^8 (coeffs for C00..C13)
    
    Constraints:
    - Entries should be in {-1, 0, 1} or {-0.5, 0, 0.5, 1, 1.5, 2} etc.
    - To keep search space manageable, we restrict to integers {-1, 0, 1} initially.
    - We use a randomized local search / simulated annealing style approach.
    """
    n, m, p = 2, 2, 4
    rank = 12
    
    # Define the target tensor T[i,j,k] = 1 for all i,j,k.
    # Actually, the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Initialize with a random sparse configuration
    # We will try to optimize the "error" sum (sum of squared errors) to 0.
    
    # Since the space of {-1,0,1}^36 (for 12 terms * 3 vectors * 4-8 dims) is huge,
    # we use a heuristic:
    # 1. Start with a valid rank-14 solution.
    # 2. Try to merge or eliminate terms to reduce rank to 12.
    #    This is hard directly.
    
    # Alternative: Randomized restarts with local search.
    # Let's try a specific structural hypothesis:
    # Maybe the rank-12 solution involves terms that mix the two blocks.
    
    # For now, let's implement a simple random search over a restricted subspace.
    # We will represent U, V, W as lists of rank vectors.
    
    # To make it feasible within time limits, we'll try a few specific patterns.
    # Pattern: 6 terms for the "left" part of A and 6 for the "right"? No.
    
    # Let's try to find a solution where U has at most 2 non-zeros per row, etc.
    
    # Given the complexity, if we don't find one quickly, we return None.
    # We will use a time-boxed search.
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for this search
    
    # Heuristic: Start from Strassen rank-14 and try to drop 2 terms, adjusting others.
    # This is a local search in the space of decompositions.
    
    # For this specific task, let's try a different known construction if available.
    # Actually, let's just return None to indicate no better solution found yet,
    # and rely on the fallback.
    
    # However, the directive asks for a STRUCTURAL change.
    # Let's implement a simple genetic algorithm or random local search for rank 12.
    
    # Represent state as a list of 12 tuples (u, v, w)
    # u is length 4, v is length 8, w is length 8.
    
    def random_state():
        state = []
        for _ in range(rank):
            u = [random.choice([-1, 0, 1]) for _ in range(4)]
            v = [random.choice([-1, 0, 1]) for _ in range(8)]
            w = [random.choice([-1, 0, 1]) for _ in range(8)]
            state.append((u, v, w))
        return state

    def evaluate(state):
        # Calculate error
        error = 0.0
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        for r in range(rank):
            u, v, w = state[r]
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        # Compute tensor product
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: U index is i*m+j = 2i+j. V index is j*p+k = 2j+k. W index is k*n+i = 2k+i.
        
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    u_idx = 2*i + j
                    v_idx = 2*j + k
                    w_idx = 2*k + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    error += (s - 1.0)**2
        return error

    def mutate(state):
        new_state = [list(s) for s in state]
        r = random.randint(0, rank-1)
        vec_idx = random.randint(0, 2) # 0:u, 1:v, 2:w
        if vec_idx == 0:
            dim = 4
        elif vec_idx == 1:
            dim = 8
        else:
            dim = 8
            
        pos = random.randint(0, dim-1)
        
        # Change value to a different integer in {-1, 0, 1}
        old_val = new_state[r][vec_idx][pos]
        choices = [x for x in [-1, 0, 1] if x != old_val]
        new_val = random.choice(choices)
        new_state[r][vec_idx][pos] = new_val
        
        return [tuple(s) for s in new_state]

    # Local search
    best_state = None
    best_err = float('inf')
    
    for restart in range(20):
        if time.time() > deadline:
            break
        state = random_state()
        err = evaluate(state)
        if err < best_err:
            best_err = err
            best_state = state
            if err == 0:
                break
        
        # Local search
        for step in range(100):
            if time.time() > deadline:
                break
            new_state = mutate(state)
            new_err = evaluate(new_state)
            if new_err < err:
                state = new_state
                err = new_err
                if err < best_err:
                    best_err = err
                    best_state = state
                    if err == 0:
                        break
            if err == 0:
                break
        if best_err == 0:
            break

    if best_err == 0:
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        for r in range(rank):
            u, v, w = best_state[r]
            for idx in range(4):
                U[idx][r] = u[idx]
            for idx in range(8):
                V[idx][r] = v[idx]
            for idx in range(8):
                W[idx][r] = w[idx]
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = 2, 2, 4
    
    # 1. Try to find a rank-12 solution via structured search
    # Note: The search is time-boxed. If it fails, we fall back.
    sol_12 = _search_rank_12()
    if sol_12 is not None:
        if _verify_decomposition(sol_12['U'], sol_12['V'], sol_12['W'], n, m, p, tol=1e-9):
            return sol_12

    # 2. Fallback to rank-14 Strassen
    sol = _strassen_2x2x4()
    if _verify_decomposition(sol['U'], sol['V'], sol['W'], n, m, p, tol=1e-9):
        return sol

    # 3. Fallback to trivial rank-16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
