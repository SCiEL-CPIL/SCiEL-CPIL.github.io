"""Rewritten program for matrix multiplication 3x3 * 3x7 rank search."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 7]

# Deadline guard: kill at 150s wall-clock, keep a safety margin.
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _tensor_index(n, m, p):
    """Yield (i,j,k) indices for the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                yield (i, j, k)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                val = np.sum(U[row_u, :] * V[row_v, :] * W[row_w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if i_a != i_c or k_b != k_c:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
                else:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_3x3x3(U, V, W):
    """Verify 3x3x3 decomposition."""
    return _verify(U, V, W, 3, 3, 3)

def _search_3x3x3(rank, time_limit=10.0, seed=42):
    """Search for a rank decomposition of 3x3x3."""
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try integer/half-integer values
    candidates = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    
    for restart in range(50):
        if time.time() > time_limit:
            break
            
        U = rng.choice(candidates, size=(9, rank))
        V = rng.choice(candidates, size=(9, rank))
        W = rng.choice(candidates, size=(9, rank))
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        c = k * 3 + i
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        err += (val - 1.0) ** 2
            for a in range(9):
                i_a = a // 3
                j_a = a % 3
                for b in range(9):
                    j_b = b // 3
                    k_b = b % 3
                    if j_a != j_b:
                        continue
                    for c in range(9):
                        k_c = c // 3
                        i_c = c % 3
                        if i_a != i_c or k_b != k_c:
                            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                            err += val ** 2
            return err
        
        err = compute_error(U, V, W)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error < 1e-12:
                return best_U, best_V, best_W
        
        # Local search: random coordinate descent
        for step in range(1000):
            if time.time() > time_limit:
                break
            
            # Pick a random matrix and element
            mat_idx = rng.randint(3)
            row, col = rng.randint(9), rng.randint(rank)
            
            old_val = [U, V, W][mat_idx][row, col]
            new_val = rng.choice(candidates)
            
            if mat_idx == 0:
                U[row, col] = new_val
            elif mat_idx == 1:
                V[row, col] = new_val
            else:
                W[row, col] = new_val
            
            new_err = compute_error(U, V, W)
            
            if new_err < err:
                err = new_err
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-12:
                        return best_U, best_V, best_W
            else:
                # Revert
                if mat_idx == 0:
                    U[row, col] = old_val
                elif mat_idx == 1:
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val
                    
    if best_error < 1e-12:
        return best_U, best_V, best_W
    return None

def _try_random_search(n, m, p, rank, max_iters=5000, seed=42):
    """Try to find a decomposition of the given rank using random initialization + local search."""
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    U0, V0, W0 = _schoolbook(n, m, p)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Simple gradient-free local search: perturb and keep if error decreases
        # Compute error
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        err += (val - 1.0) ** 2
            # Check zeros
            for a in range(n * m):
                i_a = a // m
                j_a = a % m
                for b in range(m * p):
                    j_b = b // p
                    k_b = b % p
                    if j_a != j_b:
                        continue
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        if i_a != i_c or k_b != k_c:
                            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                            err += val ** 2
            return err
        
        err = compute_error(U, V, W)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error < 1e-12:
                break
        
        # Perturbation
        if it % 10 == 0:
            # Larger step
            U += 0.5 * rng.randn(n * m, rank)
            V += 0.5 * rng.randn(m * p, rank)
            W += 0.5 * rng.randn(p * n, rank)
        else:
            U += 0.1 * rng.randn(n * m, rank)
            V += 0.1 * rng.randn(m * p, rank)
            W += 0.1 * rng.randn(p * n, rank)
    
    if best_error < 1e-12:
        # Snap to integers/half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        if _verify(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    return None

def _try_lp_rank(n, m, p, rank):
    """
    Try to find a decomposition of the given rank using linear programming.
    For a fixed rank, the equations are linear in U, V, W if we fix one factor.
    We can fix W to have orthonormal columns (or some simple structure) and solve for U, V.
    Actually, this is a 3-linear problem. Let's use a different approach.
    
    Fix V to be a simple matrix (e.g., V = I_padded or random with full column rank).
    Then the equations become bilinear in U and W.
    
    Alternatively, we can use the fact that for rank r, we need:
    sum_c U[a,c] V[b,c] W[c',c] = T[a,b,c']
    
    This is equivalent to: for each c', U * (V * W[:, c']^T)^T ... no, this is 3-tensor.
    
    Let's try: fix W to be a random matrix with full column rank.
    Then for each c' (row of W), we have:
    sum_c U[a,c] V[b,c] W[c',c] = T[a,b,c']
    
    This is: (U * diag(W[c',:]) * V^T)[a,b] = T[a,b,c']
    
    So if we let X = U * diag(W[c',:]) and Y = V^T, then X * Y = T[:, :, c']
    
    This is a matrix factorization problem: X * Y = M_c' where X is (n*m) x r, Y is r x (m*p).
    
    If we fix W, we need to find U and V such that for all c':
    U * diag(W[c',:]) * V^T = T[:, :, c']
    
    Let D_c' = diag(W[c',:]). Then U * D_c' * V^T = T_c'.
    
    If W has full column rank, we can set V = W^{-1} (if W is square, but it's not).
    
    This is getting complicated. Let's use a simpler approach: try small ranks with integer/half-integer entries.
    """
    return None

def _try_integer_search(n, m, p, rank, max_attempts=10000, seed=42):
    """
    Search for decompositions with entries in {0, 1/2, 1, 3/2, 2} or similar small sets.
    This is a combinatorial search, so limit the scope.
    """
    if rank > 15:
        return None  # Too large for brute force
    
    rng = np.random.RandomState(seed)
    candidates = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    # This is infeasible for rank > 5 or so. Just try a few random ones.
    for _ in range(min(max_attempts, 100)):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        U = rng.choice(candidates, size=(n*m, rank))
        V = rng.choice(candidates, size=(m*p, rank))
        W = rng.choice(candidates, size=(p*n, rank))
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: Decompose 3x3x7 into two 3x3x3 and one 3x3x1.
    # If we find a rank R decomposition for 3x3x3, the total rank is 2*R + 9.
    # Schoolbook 3x3x3 is 27. 2*27 + 9 = 63.
    # If we find R=26, total is 61.
    # If we find R=23, total is 55.
    
    # Search for 3x3x3 decomposition
    found_3x3x3 = None
    for r in range(23, 27):
        if time.time() > _DEADLINE - 30: # Reserve time for final check
            break
        res = _search_3x3x3(r, time_limit=10.0, seed=r)
        if res is not None:
            found_3x3x3 = res
            break
            
    if found_3x3x3 is not None:
        U_3, V_3, W_3 = found_3x3x3
        r_3 = U_3.shape[1]
        total_rank = 2 * r_3 + 9
        
        if total_rank < best_rank:
            # Construct the 3x3x7 decomposition
            # U_total: 9 x total_rank
            # V_total: 21 x total_rank
            # W_total: 21 x total_rank
            
            U_total = np.zeros((9, total_rank))
            V_total = np.zeros((21, total_rank))
            W_total = np.zeros((21, total_rank))
            
            # Block 1: A * B1 (3x3 * 3x3)
            # U_3 is 9 x r_3
            # V_3 is 9 x r_3 (corresponds to B1)
            # W_3 is 9 x r_3 (corresponds to C1)
            
            U_total[:, 0:r_3] = U_3
            # V_3 corresponds to B1 (rows 0-8 of V_total, which is B1 part of B)
            V_total[0:9, 0:r_3] = V_3
            # W_3 corresponds to C1 (rows 0-8 of W_total? No. W is indexed by k*n+i.
            # For 3x3x3, W is 9x r.
            # In 3x3x7, W is 21x total_rank.
            # The first 9 rows of W correspond to k=0,1,2?
            # W index: k*n + i.
            # For B1 (cols 0-2 of B), k is 0-2.
            # So W rows 0-8 correspond to C1.
            W_total[0:9, 0:r_3] = W_3
            
            # Block 2: A * B2 (3x3 * 3x3)
            # B2 is cols 3-5 of B.
            # V rows for B2: j*p + k.
            # j=0, k=3,4,5 -> 0, 1, 2? No.
            # V index: j*p + k.
            # B is 3x7.
            # B1 is cols 0,1,2. B2 is cols 3,4,5. B3 is col 6.
            # V rows:
            # j=0: k=0..6 -> 0..6
            # j=1: k=0..6 -> 7..13
            # j=2: k=0..6 -> 14..20
            # 
            # For B1 (k=0,1,2):
            # j=0: 0,1,2
            # j=1: 7,8,9
            # j=2: 14,15,16
            # 
            # My V_3 from 3x3x3 search assumes B is 3x3.
            # So V_3 rows correspond to j=0,1,2 and k=0,1,2.
            # V_3 row 0 -> j=0, k=0. In 3x3x7, this is V row 0.
            # V_3 row 1 -> j=0, k=1. In 3x3x7, this is V row 1.
            # V_3 row 2 -> j=0, k=2. In 3x3x7, this is V row 2.
            # V_3 row 3 -> j=1, k=0. In 3x3x7, this is V row 7.
            # V_3 row 4 -> j=1, k=1. In 3x3x7, this is V row 8.
            # V_3 row 5 -> j=1, k=2. In 3x3x7, this is V row 9.
            # V_3 row 6 -> j=2, k=0. In 3x3x7, this is V row 14.
            # V_3 row 7 -> j=2, k=1. In 3x3x7, this is V row 15.
            # V_3 row 8 -> j=2, k=2. In 3x3x7, this is V row 16.
            
            # So we need to map V_3 rows to V_total rows.
            # Mapping:
            # v3_row = j*3 + k (for k in 0..2)
            # v7_row = j*7 + k
            
            v3_to_v7 = []
            for j in range(3):
                for k in range(3):
                    v3_row = j * 3 + k
                    v7_row = j * 7 + k
                    v3_to_v7.append((v3_row, v7_row))
            
            # Block 1 (B1, k=0..2)
            for v3_r, v7_r in v3_to_v7:
                V_total[v7_r, 0:r_3] = V_3[v3_r, :]
                
            # W mapping for Block 1 (C1, k=0..2)
            # W index: k*n + i.
            # W_3 row: k*3 + i.
            # W_total row: k*3 + i (since n=3, and k is 0..2, these are rows 0..8).
            # So W_total[0:9, 0:r_3] = W_3 is correct.
            
            # Block 2 (B2, k=3..5)
            # U is same.
            U_total[:, r_3:2*r_3] = U_3
            
            # V mapping for B2
            # B2 cols are 3,4,5.
            # v3_row = j*3 + k_local (k_local 0..2)
            # v7_row = j*7 + (k_local + 3)
            
            v3_to_v7_b2 = []
            for j in range(3):
                for k_local in range(3):
                    v3_row = j * 3 + k_local
                    v7_row = j * 7 + (k_local + 3)
                    v3_to_v7_b2.append((v3_row, v7_row))
            
            for v3_r, v7_r in v3_to_v7_b2:
                V_total[v7_r, r_3:2*r_3] = V_3[v3_r, :]
                
            # W mapping for C2 (k=3..5)
            # C2 is cols 3,4,5 of C.
            # W index: k*n + i.
            # k is 3,4,5.
            # W_total rows: 3*3 + i = 9+i, 4*3+i=12+i, 5*3+i=15+i.
            # W_3 rows: k_local*3 + i.
            
            w3_to_w7_b2 = []
            for k_local in range(3):
                for i in range(3):
                    w3_row = k_local * 3 + i
                    w7_row = (k_local + 3) * 3 + i
                    w3_to_w7_b2.append((w3_row, w7_row))
            
            for w3_r, w7_r in w3_to_w7_b2:
                W_total[w7_r, r_3:2*r_3] = W_3[w3_r, :]
                
            # Block 3 (B3, k=6)
            # Schoolbook for 3x3x1.
            # Rank 9.
            # U: 9x9. U[i*3+j, col] = 1 if col == i*3+j?
            # For 3x3x1:
            # U row i*3+j, V row j*1+0 = j, W row 0*3+i = i.
            # Wait, V is 3x1. V rows are j.
            # W is 1x3. W rows are i.
            
            # In our 3x3x7 setup:
            # V row for B3 (k=6): j*7 + 6.
            # W row for C3 (k=6): 6*3 + i = 18+i.
            
            for i in range(3):
                for j in range(3):
                    col = 2*r_3 + (i*3 + j)
                    U_total[i*3 + j, col] = 1
                    V_total[j*7 + 6, col] = 1
                    W_total[18 + i, col] = 1
            
            # Verify
            if _verify(U_total, V_total, W_total, 3, 3, 7):
                best_rank = total_rank
                best_U, best_V, best_W = U_total, V_total, W_total
            else:
                print("Constructed solution failed verification")
                
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
