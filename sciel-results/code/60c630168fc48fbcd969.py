"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0
                for c in range(rank):
                    lhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if lhs != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            # Row i*m+j of U, row j*p+k of V for all k
            u_row = U[i * m + j, :]  # (rank,)
            # V rows: j*p to j*p+p-1
            v_rows = V[j * p : (j + 1) * p, :]  # (p, rank)
            # W rows: k*n + i for k in range(p)
            w_vals = np.array([W[k * n + i, :] for k in range(p)])  # (p, rank)
            # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            for k in range(p):
                s = np.sum(u_row * v_rows[k, :] * w_vals[k, :])
                if s != 1:
                    return False
    return True

def _round_to_halfint(arr, tol=1e-6):
    """Round array entries to nearest half-integer or integer if within tol."""
    result = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        # Try rounding to nearest 0.5
        rounded = np.round(val * 2) / 2.0
        if abs(val - rounded) < tol:
            result[idx] = rounded
        else:
            return None
    return result

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions."""
    result = []
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        # Check if it's a half-integer
        if abs(val * 2 - round(val * 2)) < 1e-9:
            result.append(Fraction(int(round(val * 2)), 2))
        else:
            result.append(Fraction(val).limit_denominator(1000000))
    return result

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) with entries in {-1, 0, 1}.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U: (i,j) -> coefficients for A
    # V: (j,k) -> coefficients for B  
    # W: (k,i) -> coefficients for C reconstruction
    
    # For Strassen, we have:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = -1 # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[2, 3] = -1 # B10
    W[0, 3] = -1 # C00
    W[1, 3] = 1  # C01
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = 1  # C00
    W[2, 4] = 1  # C10
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[2, 5] = 1  # B10
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[1, 6] = 1  # B01
    V[3, 6] = 1  # B11
    W[0, 6] = -1 # C00
    
    return U, V, W

def _embed_2x4x5():
    """Embed the 2x2x2 Strassen decomposition into 2x4x5.
    
    We view the 2x4x5 multiplication as:
    - First dimension: 2 (same as 2x2x2)
    - Second dimension: 4 = 2*2 (two blocks of size 2)
    - Third dimension: 5 = 2*2 + 1 (two blocks of size 2, plus 1 extra)
    
    Strategy: Use the Strassen 7-term decomposition for the 2x2x2 core,
    and extend to handle the 4 and 5 dimensions.
    
    For the m=4 dimension: split into two 2x2 blocks
    For the p=5 dimension: split into two 2x2 blocks + 1x1
    
    Total rank: 7 * 2 * 2 + 7 * 2 * 1 + 7 * 1 * 1 = 28 + 14 + 7 = 49? No...
    
    Actually, let me think more carefully. We need to decompose:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..4
    
    Split j into j1 in 0..1 and j2 in 2..3:
    C[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2+2]*B[j2+2,k]
    
    Split k into k1 in 0..1 and k2 in 2..3 and k3=4:
    C[i,k1] = sum_{j1} A[i,j1]*B[j1,k1] + sum_{j2} A[i,j2+2]*B[j2+2,k1]
    C[i,k2+2] = sum_{j1} A[i,j1]*B[j1,k2+2] + sum_{j2} A[i,j2+2]*B[j2+2,k2+2]
    C[i,4] = sum_{j1} A[i,j1]*B[j1,4] + sum_{j2} A[i,j2+2]*B[j2+2,4]
    
    Each 2x2x2 sub-problem can use Strassen rank 7.
    
    For C[i,k1]: two 2x2x2 problems (j1xk1 and j2xk1), each rank 7 -> total 14
    For C[i,k2+2]: two 2x2x2 problems, each rank 7 -> total 14
    For C[i,4]: two 2x2x1 problems. A 2x2x1 is just a 2x2 matrix times a vector.
       This has rank at most 4 (schoolbook) or possibly less.
    
    Hmm, this is getting complicated. Let me try a different approach.
    
    Alternative: Direct construction using the fact that we can embed smaller decompositions.
    
    Actually, let me just try the direct ALS approach but with better initialization
    based on the Strassen structure.
    """
    pass

def construct():
    """Find rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40
    
    # Start with schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try structured Strassen-based approach first
    # For 2x4x5, we can try rank 35 = 7 * 5 (Strassen 7 for 2x2x2, extended)
    # Or rank 28 = 7 * 4
    
    # Let's try a direct approach: use the structure of the problem
    # The key insight: 2x4x5 can be seen as combining 2x2x2 blocks
    
    # First, try rank 35 with structured initialization
    target_rank = 35
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Use Strassen 2x2x2 as base and extend
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # For 2x4x5:
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # 
    # Split j into two 2-blocks: j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}
    # Split k into: k = k1*2 + k2 where k1 in {0,1,2}, k2 in {0,1} for k<4, and k=4 special
    #
    # Actually, let's use a simpler embedding:
    # Treat the 4 as 2+2 and the 5 as 2+2+1
    # 
    # For each combination of (j_block, k_block), we have a 2x2x2 sub-problem
    # j_block in {0,1}, k_block in {0,1,2} (where k_block=2 is the single k=4)
    #
    # For k_block=0 (k=0,1): 2x2x2 -> rank 7
    # For k_block=1 (k=2,3): 2x2x2 -> rank 7  
    # For k_block=2 (k=4): 2x2x1 -> rank at most 4, but let's use 7 for simplicity
    #
    # Total: 2 * 7 + 2 * 7 + 2 * 7 = 42? That's worse than schoolbook.
    #
    # Let me try a different decomposition.
    # 
    # Actually, for 2x4x5, the rank should be at most:
    # rank(2x4x5) <= rank(2x2x5) + rank(2x2x5) 
    # And rank(2x2x5) <= rank(2x2x2) + rank(2x2x2) + rank(2x2x1)
    # = 7 + 7 + 4 = 18? 
    # So rank(2x4x5) <= 2 * 18 = 36? Still close to 40.
    #
    # Let me just do the ALS search with better initialization.
    
    # Better initialization: use partial Strassen structure
    np.random.seed(42)
    
    # Initialize with a mix of structured and random
    for c in range(target_rank):
        # First 14 columns: Strassen for j-blocks 0,1 and k-blocks 0,1
        if c < 14:
            block = c // 7  # 0 or 1
            term = c % 7     # 0-6
    
            # j_block = 0: j in {0,1}, j_block = 1: j in {2,3}
            # k_block = 0: k in {0,1}, k_block = 1: k in {2,3}
            jb = c // 7
            kb = c % 2  # This doesn't work well, let me restructure
            
        # Just use random for now, but with some structure
        U[:, c] = np.random.randn(n * m) * 0.5
        V[:, c] = np.random.randn(m * p) * 0.5
        W[:, c] = np.random.randn(p * n) * 0.5
    
    # ALS refinement
    converged = False
    for iteration in range(300):
        if time.time() > deadline - 5.0:
            break
        
        # Fix V and W, solve for U
        U_new = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
    
    # Check if rank 35 works
    if _verify_fast(U, V, W, n, m, p):
        U_rounded = _round_to_halfint(U, tol=1e-4)
        V_rounded = _round_to_halfint(V, tol=1e-4)
        W_rounded = _round_to_halfint(W, tol=1e-4)
        
        if U_rounded is not None and V_rounded is not None and W_rounded is not None:
            if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            else:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
        else:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
    
    # If rank 35 didn't work, try other ranks
    if best_rank == schoolbook_rank:
        for target_rank in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
            if time.time() > deadline - 10.0:
                break
            
            max_restarts = 10
            found = False
            
            for restart in range(max_restarts):
                if time.time() > deadline - 5.0:
                    break
                
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
                
                for iteration in range(200):
                    if time.time() > deadline - 3.0:
                        break
                    
                    U_new = np.zeros((n * m, target_rank))
                    for i in range(n):
                        for j in range(m):
                            A = np.zeros((p, target_rank))
                            b = np.ones(p)
                            for k in range(p):
                                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                U_new[i * m + j, :] = x
                            except:
                                U_new[i * m + j, :] = U[i * m + j, :]
                    
                    V_new = np.zeros((m * p, target_rank))
                    for j in range(m):
                        for k in range(p):
                            A = np.zeros((n, target_rank))
                            b = np.ones(n)
                            for i in range(n):
                                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                V_new[j * p + k, :] = x
                            except:
                                V_new[j * p + k, :] = V[j * p + k, :]
                    
                    W_new = np.zeros((p * n, target_rank))
                    for k in range(p):
                        for i in range(n):
                            A = np.zeros((m, target_rank))
                            b = np.ones(m)
                            for j in range(m):
                                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                W_new[k * n + i, :] = x
                            except:
                                W_new[k * n + i, :] = W[k * n + i, :]
                    
                    U, V, W = U_new, V_new, W_new
                
                if _verify_fast(U, V, W, n, m, p):
                    U_rounded = _round_to_halfint(U, tol=1e-4)
                    V_rounded = _round_to_halfint(V, tol=1e-4)
                    W_rounded = _round_to_halfint(W, tol=1e-4)
                    
                    if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                        if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                            best_rank = target_rank
                            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        else:
                            best_rank = target_rank
                            best_U, best_V, best_W = U, V, W
                    else:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                    found = True
                    break
            
            if found:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Try to convert to fractions for cleanliness
    try:
        U_frac = []
        for row in U_list:
            U_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        V_frac = []
        for row in V_list:
            V_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        W_frac = []
        for row in W_list:
            W_frac.append([Fraction(str(x)).limit_denominator(1000000) for x in row])
        
        # Verify with fractions
        if _verify(U_frac, V_frac, W_frac, n, m, p):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    except:
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
