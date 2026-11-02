"""Matrix multiplication 2,2,5 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def _build_target(n, m, p):
    """Construct the target tensor T[i, j, k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the target tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    T_comp = np.zeros((n, m, p), dtype=np.float64)
    
    # Vectorized computation for speed
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We can loop over c and use broadcasting or explicit loops for small dims.
    # Given small dims (2,2,5), explicit loops over c and vectorized over i,j,k is fine.
    
    for c in range(rank):
        u_c = U[:, c]
        v_c = V[:, c]
        w_c = W[:, c]
        
        # Reshape to (n, m), (m, p), (p, n)
        u_mat = u_c.reshape(n, m)
        v_mat = v_c.reshape(m, p)
        w_mat = w_c.reshape(p, n)
        
        # We need T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is not a standard matrix product directly due to the index cycling.
        # Let's do it with a small loop over k or i to be safe and fast enough.
        # Actually, for n=2, m=2, p=5, the tensor is small (20 elements).
        for i in range(n):
            for j in range(m):
                # u_val = u_mat[i, j]
                # If u_val is 0, skip?
                if u_mat[i, j] == 0: continue
                for k in range(p):
                    T_comp[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T_comp, T_target, atol=tol)

def _build_rank19(n, m, p):
    """
    Construct a rank-19 decomposition.
    Split p=5 into p1=2 and p2=3.
    Part 1: 2x2x2 using Strassen (Rank 7).
    Part 2: 2x2x3 using Schoolbook (Rank 12).
    Total Rank: 19.
    """
    rank1 = 7
    rank2 = 12
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # --- Part 1: Strassen for columns 0, 1 ---
    # n=2, m=2, p_part=2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Standard Strassen for 2x2 matrices
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    
    U_s = np.zeros((4, 7), dtype=np.float64)
    V_s = np.zeros((4, 7), dtype=np.float64)
    W_s = np.zeros((4, 7), dtype=np.float64)
    
    # Indices for 2x2:
    # U: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # p1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[1,0]=1; V_s[2,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    
    # p2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1
    
    # p3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    
    # p4
    U_s[3,3]=1
    V_s[0,3]=-1; V_s[2,3]=1
    W_s[0,3]=1; W_s[2,3]=1
    
    # p5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    
    # p6
    U_s[0,5]=-1; U_s[2,5]=1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    
    # p7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1
    
    # Copy Strassen to main arrays for cols 0,1
    U[:, :7] = U_s
    # V: cols 0,1 of B correspond to V indices (j*5+k) for k=0,1
    # V_s indices are (j*2+k). We need to map (j, k) -> j*5+k
    for j in range(2):
        for k in range(2):
            V[j*5+k, :7] = V_s[j*2+k, :7]
    # W: cols 0,1 of C correspond to W indices (k*2+i) for k=0,1
    # W_s indices are (k*2+i).
    for k in range(2):
        for i in range(2):
            W[k*2+i, :7] = W_s[k*2+i, :7]
    
    # --- Part 2: Schoolbook for columns 2, 3, 4 ---
    # n=2, m=2, p_part=3
    # Rank 12
    # U_sc: 4x12, V_sc: 6x12, W_sc: 6x4
    U_sc = np.zeros((4, 12), dtype=np.float64)
    V_sc = np.zeros((6, 12), dtype=np.float64)
    W_sc = np.zeros((6, 4), dtype=np.float64) # Wait, W size is p*n = 3*2=6? No, W is (p*n, rank).
    # W shape is (p*n, rank). For p=3, n=2, W is 6x12.
    W_sc = np.zeros((6, 12), dtype=np.float64)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_sc[i*2+j, col] = 1
                V_sc[j*3+k, col] = 1
                W_sc[k*2+i, col] = 1
                col += 1
                
    # Copy Schoolbook to main arrays for cols 2,3,4
    offset = 7
    U[:, 7:19] = U_sc
    # V: cols 2,3,4 of B. V indices (j*5+k) for k=2,3,4.
    # V_sc indices are (j*3+k) for k=0,1,2.
    for j in range(2):
        for k in range(3):
            V[j*5+(k+2), 7:19] = V_sc[j*3+k, 7:19] # Note: V_sc only has 12 cols, so 7:19 is out of bounds for V_sc?
            # V_sc is 6x12. We are copying to V which is 10x19.
            # V_sc[:, r] goes to V[j*5+k+2, 7+r]
            pass # The loop above was pseudo-code. Let's do it properly.
            
    # Redo copy for Part 2
    for r in range(12):
        for i in range(2):
            for j in range(2):
                U[i*2+j, 7+r] = U_sc[i*2+j, r]
        for j in range(2):
            for k in range(3):
                V[j*5+(k+2), 7+r] = V_sc[j*3+k, r]
        for k in range(3):
            for i in range(2):
                W[(k+2)*2+i, 7+r] = W_sc[k*2+i, r]
                
    return U, V, W

def _als_solve_U(V, W, T, n, m, p, rank):
    """
    Solve for U given V, W, T.
    T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    For fixed i,j:
    T[i,j,:] (vector of size p) = sum_c U[i,j,c] * (V[:,c] * W[:,c] permuted?)
    
    Let's vectorize.
    We want to find U (n*m, rank) such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    For a fixed (i,j), let u_vec = U[i*m+j, :].
    The equation is:
    sum_c u_vec[c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k] for all k.
    
    Let M_ij be a matrix of shape (p, rank) where:
    M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
    
    Then M_ij @ u_vec = T[i,j,:]
    
    We solve this least squares problem for each (i,j).
    """
    U = np.zeros((n*m, rank), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            # Construct M_ij
            # k from 0 to p-1
            # c from 0 to rank-1
            M_ij = np.zeros((p, rank), dtype=np.float64)
            for k in range(p):
                # V index: j*p + k
                v_row = V[j*p+k, :]
                # W index: k*n + i
                w_row = W[k*n+i, :]
                M_ij[k, :] = v_row * w_row
            
            # Solve M_ij @ u_vec = T[i,j,:]
            # Use lstsq
            try:
                u_vec, residuals, rank_mat, sv = np.linalg.lstsq(M_ij, T[i,j,:], rcond=None)
                U[i*m+j, :] = u_vec
            except np.linalg.LinAlgError:
                U[i*m+j, :] = 0
                
    return U

def _als_solve_V(U, W, T, n, m, p, rank):
    """
    Solve for V given U, W, T.
    For fixed (j,k):
    T[:,j,k] (vector of size n) = sum_c V[j*p+k, c] * (U[:,c] * W[:,c] permuted?)
    
    Equation:
    sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = T[i,j,k] for all i.
    
    Let N_jk be a matrix of shape (n, rank) where:
    N_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
    
    Then N_jk @ v_vec = T[:,j,k]
    """
    V = np.zeros((m*p, rank), dtype=np.float64)
    for j in range(m):
        for k in range(p):
            N_jk = np.zeros((n, rank), dtype=np.float64)
            for i in range(n):
                # U index: i*m + j
                u_row = U[i*m+j, :]
                # W index: k*n + i
                w_row = W[k*n+i, :]
                N_jk[i, :] = u_row * w_row
            
            try:
                v_vec, residuals, rank_mat, sv = np.linalg.lstsq(N_jk, T[:,j,k], rcond=None)
                V[j*p+k, :] = v_vec
            except np.linalg.LinAlgError:
                V[j*p+k, :] = 0
                
    return V

def _als_solve_W(U, V, T, n, m, p, rank):
    """
    Solve for W given U, V, T.
    For fixed (k,i):
    T[i,:,k] (vector of size m) = sum_c W[k*n+i, c] * (U[:,c] * V[:,c] permuted?)
    
    Equation:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] for all j.
    
    Let P_ki be a matrix of shape (m, rank) where:
    P_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
    
    Then P_ki @ w_vec = T[i,:,k]
    """
    W = np.zeros((p*n, rank), dtype=np.float64)
    for k in range(p):
        for i in range(n):
            P_ki = np.zeros((m, rank), dtype=np.float64)
            for j in range(m):
                # U index: i*m + j
                u_row = U[i*m+j, :]
                # V index: j*p + k
                v_row = V[j*p+k, :]
                P_ki[j, :] = u_row * v_row
            
            try:
                w_vec, residuals, rank_mat, sv = np.linalg.lstsq(P_ki, T[i,:,k], rcond=None)
                W[k*n+i, :] = w_vec
            except np.linalg.LinAlgError:
                W[k*n+i, :] = 0
                
    return W

def _try_rank_als(U_init, V_init, W_init, T, n, m, p, rank, max_iters=100, tol=1e-10):
    """
    Run ALS starting from initialization.
    """
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    # Trim to target rank if init is larger (should not happen if we init correctly)
    if U.shape[1] > rank:
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
        
    for it in range(max_iters):
        U_new = _als_solve_U(V, W, T, n, m, p, rank)
        V_new = _als_solve_V(U_new, W, T, n, m, p, rank)
        W_new = _als_solve_W(U_new, V_new, T, n, m, p, rank)
        
        # Check convergence
        diff_U = np.linalg.norm(U_new - U)
        diff_V = np.linalg.norm(V_new - V)
        diff_W = np.linalg.norm(W_new - W)
        
        U, V, W = U_new, V_new, W_new
        
        if diff_U + diff_V + diff_W < tol:
            break
            
    # Verify
    if _check_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None

def _verify_strict(U, V, W, n, m, p):
    """
    Strict verification using Fractions.
    Returns True if exact.
    """
    # Convert to Fractions
    U_f = [[Fraction(x).limit_denominator(100) for x in row] for row in U]
    V_f = [[Fraction(x).limit_denominator(100) for x in row] for row in V]
    W_f = [[Fraction(x).limit_denominator(100) for x in row] for row in W]
    
    rank = len(U_f[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                if s != 1:
                    return False
    return True

def _make_decomposition(U, V, W):
    """
    Convert numpy arrays to list of lists of integers/half-integers.
    """
    def clean_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                # val is float
                # Nearest half integer is round(2*val)/2
                half_int = round(val * 2) / 2.0
                # Check if it's an integer
                if half_int.is_integer():
                    new_row.append(int(half_int))
                else:
                    new_row.append(Fraction(int(round(val * 2)), 2))
            result.append(new_row)
        return result
        
    return {"U": clean_matrix(U), "V": clean_matrix(V), "W": clean_matrix(W)}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    T = _build_target(n, m, p)
    
    # 1. Build Rank 19 base
    U19, V19, W19 = _build_rank19(n, m, p)
    
    # Verify Rank 19
    if _check_decomposition(U19, V19, W19, n, m, p):
        best_rank = 19
        best_result = _make_decomposition(U19, V19, W19)
    else:
        # Fallback to Schoolbook Rank 20
        rank20 = 20
        U20 = np.zeros((n*m, rank20), dtype=np.float64)
        V20 = np.zeros((m*p, rank20), dtype=np.float64)
        W20 = np.zeros((p*n, rank20), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U20[i*m+j, col] = 1
                    V20[j*p+k, col] = 1
                    W20[k*n+i, col] = 1
                    col += 1
        best_rank = 20
        best_result = _make_decomposition(U20, V20, W20)
        
    # 2. Try to improve rank using ALS
    # Target ranks: 18, 17, 16...
    # We will try to find a valid decomposition for these ranks.
    
    for target_rank in range(best_rank - 1, 5, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        found = False
        num_tries = 20  # Adjust based on time
        
        for t in range(num_tries):
            if time.time() > deadline:
                break
                
            # Initialize with noise or scaled version of higher rank
            # Simple random init
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Run ALS
            result = _try_rank_als(U_init, V_init, W_init, T, n, m, p, target_rank, max_iters=50)
            
            if result is not None:
                U_sol, V_sol, W_sol = result
                # Strict verification
                if _verify_strict(U_sol, V_sol, W_sol, n, m, p):
                    best_rank = target_rank
                    best_result = _make_decomposition(U_sol, V_sol, W_sol)
                    found = True
                    break # Found one for this rank, try next lower rank
        
        if found:
            continue # Try next lower rank
        else:
            # If we failed to find a decomposition for this rank, 
            # it might be impossible or we just didn't find it.
            # Since we want the minimal rank, if we fail at 18, we keep 19.
            # If we succeed at 18, we try 17.
            # If we fail at 17, we keep 18.
            # So we just break if we think we can't go lower?
            # No, we should try all lower ranks.
            pass

    return best_result

# EVOLVE-BLOCK-END
