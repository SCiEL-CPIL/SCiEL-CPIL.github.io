"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen's 7 bilinear forms for 2x2 * 2x2.
    Each form is (coeffs_A, coeffs_B, coeffs_C) where:
    - coeffs_A: dict mapping (i,j) -> coefficient for A[i,j]
    - coeffs_B: dict mapping (j,k) -> coefficient for B[j,k]  
    - coeffs_C: dict mapping (i,k) -> coefficient for C[i,k]
    
    Strassen:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B22-B11)
    M4 = A22*(B22-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: {(0,0): c, (1,0): c, (0,1): c, (1,1): c}
    # B_coeffs: {(0,0): c, (1,0): c, (0,1): c, (1,1): c}
    # C_coeffs: {(0,0): c, (1,0): c, (0,1): c, (1,1): c}
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):-1}  # C11 += M1, C22 += M1
    ))
    # Actually, let me be more careful. Each Strassen term M_t contributes to C via:
    # C = sum_t (linear_combination_of_M_t)
    # But for tensor decomposition, I need to express each C[i,k] as sum of products.
    # 
    # Better approach: each Strassen "multiplication" M_t = X_t * Y_t where X_t and Y_t
    # are linear combinations of A and B entries. Then C[i,k] = sum_t alpha_{t,ik} * M_t.
    # 
    # For tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_t (sum_{ij} x_{t,ij} A[i,j]) * (sum_{jk} y_{t,jk} B[j,k]) * alpha_{t,ik}
    # But this isn't quite right because M_t = X_t * Y_t is a scalar product of two vectors.
    # 
    # Actually, M_t is itself a product of two linear forms:
    # X_t = sum_{ij} x_{t,ij} A[i,j]  (scalar)
    # Y_t = sum_{jk} y_{t,jk} B[j,k]  (scalar)
    # M_t = X_t * Y_t
    # 
    # Then C[i,k] = sum_t alpha_{t,ik} * M_t = sum_t alpha_{t,ik} * X_t * Y_t
    # 
    # This gives us a tensor decomposition with rank = number of Strassen terms = 7
    # for 2x2*2x2. But wait, this only works if X_t and Y_t are scalars, which they are.
    # 
    # So for 2x2*2x2, rank 7 is achievable! Let me define the 7 terms properly.
    
    # Let me redefine. For 2x2*2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # 
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b11-b00)
    # M4 = a11*(b11-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # 
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Each M_t = X_t * Y_t where:
    # M1: X1 = a00+a11, Y1 = b00+b11
    # M2: X2 = a10+a11, Y2 = b00
    # M3: X3 = a00, Y3 = b11-b00
    # M4: X4 = a11, Y4 = b11-b00
    # M5: X5 = a00+a01, Y5 = b11
    # M6: X6 = a10-a00, Y6 = b00+b01
    # M7: X7 = a01-a11, Y7 = b10+b11
    
    # For tensor decomposition, each M_t contributes to C as:
    # C[i,k] += alpha_{t,ik} * X_t * Y_t
    # 
    # Where X_t = sum_{i',j'} x_{t,i'j'} A[i',j'] and Y_t = sum_{j',k'} y_{t,j'k'} B[j',k']
    # 
    # So the contribution of term t to C[i,k] is:
    # alpha_{t,ik} * (sum_{i'j'} x_{t,i'j'} A[i'j']) * (sum_{j'k'} y_{t,j'k'} B[j'k'])
    # 
    # This is NOT a rank-1 tensor term in general, because it's a product of two sums.
    # 
    # Wait, I think I'm confusing myself. In tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition is: C[i,k] = sum_{t=1}^r U[i,t] * V[k,t] * W[j,t] ... no.
    # 
    # The standard formulation: C[i,k] = sum_t (sum_j u_{t,ij} A[i,j]) * (sum_j v_{t,jk} B[j,k]) * w_{t,ik}
    # No, that's not right either.
    # 
    # Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # The decomposition: for all (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # This means: the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # must equal 1 for all valid (i,j,k).
    # 
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k).
    # 
    # This is a rank-r decomposition of the all-ones tensor of shape (n*m) x (m*p) x (p*n).
    # 
    # For the schoolbook: U[i*m+j, c] = 1 if c corresponds to (i,j,k), else 0.
    # 
    # For Strassen on 2x2*2x2: we need rank 7 such that for all (i,j,k) in {0,1}^3:
    # sum_{c=1}^7 U[2i+j, c] V[2j+k, c] W[2k+i, c] = 1
    # 
    # Let me construct this explicitly.
    
    # I'll define the 7 Strassen terms as triples (u_vec, v_vec, w_vec)
    # where u_vec has 4 entries (for 2x2 A), v_vec has 4 entries (for 2x2 B), w_vec has 4 entries (for 2x2 C).
    
    # Term 1: M1 = (a00+a11)*(b00+b11)
    # u1 = [1, 0, 0, 1]  (a00 + a11)
    # v1 = [1, 0, 0, 1]  (b00 + b11)
    # w1 = [1, 0, 0, -1] (contributes +1 to c00, -1 to c11)
    # Check: u1[i*m+j] * v1[j*p+k] * w1[k*n+i]
    # For (i,j,k): u1[2i+j] * v1[2j+k] * w1[2k+i]
    # (0,0,0): u1[0]*v1[0]*w1[0] = 1*1*1 = 1
    # (0,0,1): u1[0]*v1[1]*w1[2] = 1*0*0 = 0
    # (0,1,0): u1[1]*v1[2]*w1[0] = 0*0*1 = 0
    # (0,1,1): u1[1]*v1[3]*w1[2] = 0*1*0 = 0
    # (1,0,0): u1[2]*v1[0]*w1[0] = 0*1*1 = 0
    # (1,0,1): u1[2]*v1[1]*w1[2] = 0*0*0 = 0
    # (1,1,0): u1[3]*v1[2]*w1[0] = 1*0*1 = 0
    # (1,1,1): u1[3]*v1[3]*w1[3] = 1*1*(-1) = -1
    # 
    # So term 1 contributes: T[0,0,0] += 1, T[1,1,1] += -1
    # This matches: c00 gets +M1, c11 gets -M1 (from c00 = M1+M4-M5+M7, c11 = M1-M2+M3+M6)
    
    # I'll define all 7 terms and verify they sum to 1 for all (i,j,k).
    
    # This is getting complex. Let me just use the schoolbook for now and add a refinement step.
    
    return None

def _refine_decomposition(U, V, W, n, m, p, max_iter=200, time_limit=100):
    """
    Try to reduce the rank by finding linear dependencies and eliminating terms.
    Uses coordinate descent: for each term, try to express it as a combination of others.
    """
    start_time = time.time()
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    r = U.shape[1]
    
    # Try to eliminate terms one by one
    # For each term c, check if removing it can be compensated by adjusting others
    # This is a linear algebra problem
    
    # Simpler approach: use SVD-like rank reduction
    # Flatten the tensor and try low-rank approximation
    
    # For now, just verify and return
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to find a lower-rank decomposition using structured approach
    # Split into 2x2 blocks and use Strassen-like ideas
    
    # For 4x4 * 4x8:
    # A is 4x4, B is 4x8, C is 4x8
    # 
    # Block structure:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where B11, B12, B21, B22 are 2x4
    # C = [[C11, C12], [C21, C22]] where each C block is 2x4
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each product like A11*B11 is 2x2 * 2x4.
    # We can write B11 = [B11_left, B11_right] where each is 2x2.
    # Then A11*B11 = [A11*B11_left, A11*B11_right]
    # 
    # So each 2x2 * 2x4 product = two 2x2 * 2x2 products side by side.
    # Using Strassen for each 2x2*2x2 gives 7 terms each, so 14 per 2x2*2x4.
    # 
    # Total: 8 such products * 14 = 112. Still not better than 128.
    # 
    # But we can share: A11 appears in both C11 and C12 computations.
    # 
    # Let me try a different approach: direct rank reduction via optimization.
    
    # Try ranks from 127 down to 80, using alternating minimization
    best_result = (U.tolist(), V.tolist(), W.tolist())
    best_rank = n * m * p
    
    for target_rank in range(min(127, best_rank - 1), 79, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with schoolbook truncated + random perturbation
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            # Copy first target_rank terms from schoolbook
            for c in range(target_rank):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U_t[i*m+j, c] = 1.0
                V_t[j*p+k, c] = 1.0
                W_t[k*n+i, c] = 1.0
            
            # Add small random perturbation to break symmetry
            if attempt > 0:
                U_t += np.random.randn(n*m, target_rank) * 0.1
                V_t += np.random.randn(m*p, target_rank) * 0.1
                W_t += np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating minimization
            converged = False
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        if M.shape[1] >= p:
                            try:
                                U_t[i*m+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            except:
                                pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        if M.shape[1] >= n:
                            try:
                                V_t[j*p+k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            except:
                                pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        if M.shape[1] >= m:
                            try:
                                W_t[k*n+i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            except:
                                pass
                
                # Check convergence
                if _verify(U_t, V_t, W_t, n, m, p, tol=1e-8):
                    found = True
                    break
            
            if found:
                best_result = (U_t.tolist(), V_t.tolist(), W_t.tolist())
                best_rank = target_rank
                break
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
