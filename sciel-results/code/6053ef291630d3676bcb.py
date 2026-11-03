"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
import itertools
from fractions import Fraction
from collections import defaultdict

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def _validate_numpy(U, V, W, n, m, p, tol=1e-9):
    """Faster validation using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    # Build tensor T of shape (n, m, p)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This is a bit tricky. Let's use einsum.
    # U_r: (n, m, r), V_r: (m, p, r), W_r: (p, n, r)
    # We want sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(A, B):
    """
    Decompose 2x2 * 2x2 matrix multiplication using Strassen's algorithm.
    Returns the 7 scalar multiplications as a list of (U_entry, V_entry, W_entry) tuples
    where each entry is a linear combination of the input matrix elements.
    
    For A = [[a, b], [c, d]], B = [[e, f], [g, h]]:
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    C = [[m1 + m4 - m5 + m7, m3 + m5],
         [m2 + m4, m1 - m2 + m3 + m6]]
    """
    # A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # C indices: C(0,0)=m1+m4-m5+m7, C(0,1)=m3+m5
    #            C(1,0)=m2+m4, C(1,1)=m1-m2+m3+m6
    
    # For each multiplication m_r, we have:
    # U_factor: linear combo of A elements
    # V_factor: linear combo of B elements
    # W_factor: how m_r contributes to each C(i,j)
    
    # m1 = (a+d)*(e+h)
    # U: a=1, d=1
    # V: e=1, h=1
    # W: C(0,0)+=1, C(1,1)+=1
    m1_U = {(0,0): 1, (1,1): 1}
    m1_V = {(0,0): 1, (1,1): 1}
    m1_W = {(0,0): 1, (1,1): 1}
    
    # m2 = (c+d)*e
    # U: c=1, d=1
    # V: e=1
    # W: C(1,0)+=1
    m2_U = {(1,0): 1, (1,1): 1}
    m2_V = {(0,0): 1}
    m2_W = {(1,0): 1}
    
    # m3 = a*(f-h)
    # U: a=1
    # V: f=1, h=-1
    # W: C(0,1)+=1, C(1,1)+=1
    m3_U = {(0,0): 1}
    m3_V = {(0,1): 1, (1,1): -1}
    m3_W = {(0,1): 1, (1,1): 1}
    
    # m4 = d*(g-e)
    # U: d=1
    # V: g=1, e=-1
    # W: C(0,0)+=1, C(1,0)+=1
    m4_U = {(1,1): 1}
    m4_V = {(1,0): 1, (0,0): -1}
    m4_W = {(0,0): 1, (1,0): 1}
    
    # m5 = (a+b)*h
    # U: a=1, b=1
    # V: h=1
    # W: C(0,0)-=1, C(0,1)+=1
    m5_U = {(0,0): 1, (0,1): 1}
    m5_V = {(1,1): 1}
    m5_W = {(0,0): -1, (0,1): 1}
    
    # m6 = (c-a)*(e+f)
    # U: c=1, a=-1
    # V: e=1, f=1
    # W: C(1,1)+=1
    m6_U = {(1,0): 1, (0,0): -1}
    m6_V = {(0,0): 1, (0,1): 1}
    m6_W = {(1,1): 1}
    
    # m7 = (b-d)*(g+h)
    # U: b=1, d=-1
    # V: g=1, h=1
    # W: C(0,0)+=1
    m7_U = {(0,1): 1, (1,1): -1}
    m7_V = {(1,0): 1, (1,1): 1}
    m7_W = {(0,0): 1}
    
    return [
        (m1_U, m1_V, m1_W),
        (m2_U, m2_V, m2_W),
        (m3_U, m3_V, m3_W),
        (m4_U, m4_V, m4_W),
        (m5_U, m5_V, m5_W),
        (m6_U, m6_V, m6_W),
        (m7_U, m7_V, m7_W),
    ]

def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2 blocks.
    
    For 2x3 * 3x8:
    - Split A (2x3) into two 2x2 blocks? No, 3 is not even.
    - Instead, we can pad or use a different approach.
    
    Actually, let's think differently. We have n=2, m=3, p=8.
    We can view the 3x8 matrix B as having 3 rows and 8 columns.
    We can split the 8 columns into four 2-column blocks: B = [B1, B2, B3, B4] where each Bi is 3x2.
    Then C = A * B = [A*B1, A*B2, A*B3, A*B4] where each A*Bi is 2x2.
    
    For each 2x2 block product A * Bi, we can use Strassen's algorithm.
    A is 2x3, Bi is 3x2. This is not a standard 2x2 * 2x2 multiplication.
    
    Let me reconsider. The schoolbook rank is 48. Using Strassen on 2x2 blocks:
    If we could decompose into 2x2 * 2x2 blocks, each would take rank 7 instead of 4.
    But our dimensions are 2x3 and 3x8.
    
    Alternative approach: Use the fact that we can decompose the 3x8 matrix multiplication
    into smaller pieces. Let's try to use Strassen's algorithm more cleverly.
    
    Actually, let's try a different strategy. We'll use the tensor decomposition framework
    and try to find a low-rank decomposition using optimization.
    """
    pass

def _try_low_rank_decomposition(n, m, p, target_rank, max_iterations=1000, timeout=10):
    """
    Try to find a decomposition of given rank using iterative refinement.
    We use a simple approach: start with random initialization and try to minimize
    the error using gradient-free methods.
    """
    start_time = time.time()
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization
        np.random.seed(42 + attempt)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Simple iterative refinement
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            # Compute current error
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.sum((T - 1.0)**2)
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-10:
                break
            
            # Update one factor at a time
            # Update U: for each (i,j), U[i,j,:] should satisfy
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
            # This is a linear system in U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # Build matrix A where A[k, r] = sum_k' V[j,k',r] * W[k',i,r]
                    # Actually, for fixed i,j, we need:
                    # sum_r U[i,j,r] * (sum over k of V[j,k,r] * W[k,i,r]) = 1 for all k
                    # Wait, this isn't quite right. Let me re-derive.
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # For fixed i,j, this is a vector in k-space:
                    # T[i,j,:] = sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
                    # where V[j,:,r] is a vector in k-space and W[:,i,r] is a vector in k-space
                    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # This is a linear system: M[i,j] @ U[i,j,:] = ones(p)
                    # where M[i,j][k, r] = V[j,k,r] * W[k,i,r]
                    
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            M[k, r] = V[j, k, r] * W[k, i, r]
                    
                    # Solve M @ x = ones(p)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i, j, :] = x
                    except:
                        pass
            
            # Similarly update V and W
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            M[i, r] = U[i, j, r] * W[k, i, r]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j, k, :] = x
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i, j, r] * V[j, k, r]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k, i, :] = x
                    except:
                        pass
        
        if best_error < 1e-10:
            break
    
    if best_solution is not None and best_error < 1e-8:
        U, V, W = best_solution
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U.reshape(n*m, target_rank)]
        V_list = [[float(x) for x in row] for row in V.reshape(m*p, target_rank)]
        W_list = [[float(x) for x in row] for row in W.reshape(p*n, target_rank)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen's algorithm for 2x2 blocks.
    
    For 2x3 * 3x8:
    We can split the 8 columns into 4 groups of 2 columns.
    For each group, we compute A (2x3) * B_block (3x2) = C_block (2x2).
    
    The product A * B_block where A is 2x3 and B_block is 3x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    This is a 2x3 * 3x2 matrix multiplication, which has rank at most 4 (schoolbook).
    Can we do better? The tensor for 2x3 * 3x2 has dimension 2*3*2 = 12.
    The lower bound is max(2*3, 3*2, 2*2) = max(6, 6, 4) = 6.
    So we might be able to achieve rank 6 for each 2x3 * 3x2 block.
    
    If we use rank 6 for each of the 4 blocks, total rank = 4 * 6 = 24.
    But we need to be careful about how the blocks interact.
    
    Actually, since the blocks are independent (different column groups), we can
    simply concatenate the decompositions. So if each 2x3 * 3x2 block has rank r,
    the total rank is 4*r.
    
    Let's try to find a rank-6 decomposition for 2x3 * 3x2.
    """
    n_sub, m_sub, p_sub = 2, 3, 2
    
    # Try to find a rank-6 decomposition for 2x3 * 3x2
    # Theoretical lower bound is max(6, 6, 4) = 6
    # Let's try to construct one.
    
    # For 2x3 * 3x2:
    # A = [[a, b, c], [d, e, f]]
    # B = [[g, h], [i, j], [k, l]]
    # C = [[ag+bi+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    
    # We want to express this as sum of 6 rank-1 tensors.
    # This is equivalent to finding 6 triples (U_r, V_r, W_r) such that
    # C[i,j] = sum_r U_r[i, :] * V_r[:, j] * W_r[i, j]
    # where U_r is 2x3, V_r is 3x2, W_r is 2x2.
    
    # Actually, the tensor decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For n=2, m=3, p=2:
    # T[i, j, k] = sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    # Let's try a specific construction.
    # We'll use the fact that 2x3 * 3x2 can be viewed as a bilinear map.
    
    # One approach: use the "lattice" method or known decompositions.
    # For now, let's try to use optimization to find a rank-6 decomposition.
    
    result = _try_low_rank_decomposition(2, 3, 2, 6, max_iterations=2000, timeout=5)
    if result is not None:
        U_sub, V_sub, W_sub = result["U"], result["V"], result["W"]
        
        # Now extend to 2x3 * 3x8 by repeating for 4 column blocks
        n, m, p = 2, 3, 8
        rank_total = 4 * 6
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        for block in range(4):
            base = block * 6
            for i in range(n):
                for j in range(m):
                    for r in range(6):
                        U[i * m + j][base + r] = U_sub[i * m + j][r]
            for j in range(m):
                for k in range(p_sub):
                    for r in range(6):
                        V[j * p + block * p_sub + k][base + r] = V_sub[j * p_sub + k][r]
            for k in range(p_sub):
                for i in range(n):
                    for r in range(6):
                        W[(block * p_sub + k) * n + i][base + r] = W_sub[k * n + i][r]
        
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    # Schoolbook decomposition as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try Strassen-based decomposition
    if time.time() < deadline - 30:
        strassen_result = _construct_strassen_based(n, m, p)
        if strassen_result is not None:
            # Validate
            if _validate_numpy(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                rank = len(strassen_result["U"][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
    
    # Try to find even lower rank decompositions
    if time.time() < deadline - 20:
        # Try ranks from max(n*m, m*p, p*n) up to best_rank - 1
        lower_bound = max(n * m, m * p, p * n)
        for rank in range(lower_bound, best_rank):
            if time.time() > deadline - 10:
                break
            
            result = _try_low_rank_decomposition(n, m, p, rank, max_iterations=500, timeout=5)
            if result is not None:
                if _validate_numpy(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
                    break
    
    return best_result

# EVOLVE-BLOCK-END
