import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 140

def _create_target_tensor(n, m, p):
    """Create the sparse target tensor as a dense array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _reconstruct(U, V, W):
    """Reconstruct the tensor from factors."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    T_rec = _reconstruct(U, V, W)
    T_target = _create_target_tensor(n, m, p)
    return np.allclose(T_rec, T_target, atol=tol)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 matrix multiplication.
    U, V, W have shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for 2x2 matrices.
    """
    # Strassen's 7 products for 2x2:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # U: maps A entries to the 7 linear combinations
    # U[i*2+j, t] = coefficient of A[i,j] in m_t
    U = np.zeros((4, 7))
    # m1 = (a11+a22)(b11+b22) -> A part: a11 + a22
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    # m2 = (a21+a22)b11 -> A part: a21 + a22
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    # m3 = a11(b12-b22) -> A part: a11
    U[0, 2] = 1  # a11
    # m4 = a22(b21-b11) -> A part: a22
    U[3, 3] = 1  # a22
    # m5 = (a11+a12)b22 -> A part: a11 + a12
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12
    # m6 = (a21-a11)(b11+b12) -> A part: a21 - a11
    U[2, 5] = 1  # a21
    U[0, 5] = -1 # a11
    # m7 = (a12-a22)(b21+b22) -> A part: a12 - a22
    U[1, 6] = 1  # a12
    U[3, 6] = -1 # a22
    
    # V: maps B entries to the 7 linear combinations
    # V[j*2+k, t] = coefficient of B[j,k] in m_t
    V = np.zeros((4, 7))
    # m1: b11 + b22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    # m2: b11
    V[0, 1] = 1  # b11
    # m3: b12 - b22
    V[1, 2] = 1  # b12
    V[3, 2] = -1 # b22
    # m4: b21 - b11
    V[2, 3] = 1  # b21
    V[0, 3] = -1 # b11
    # m5: b22
    V[3, 4] = 1  # b22
    # m6: b11 + b12
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    # m7: b21 + b22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    
    # W: maps the 7 products to C entries
    # W[k*2+i, t] = coefficient of m_t in c_{k,i}
    W = np.zeros((4, 7))
    # c11 = m1 + m4 - m5 + m7  -> k=0, i=0 -> index 0*2+0=0
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    # c12 = m3 + m5  -> k=0, i=1 -> index 0*2+1=1
    W[1, 2] = 1
    W[1, 4] = 1
    # c21 = m2 + m4  -> k=1, i=0 -> index 1*2+0=2
    W[2, 1] = 1
    W[2, 3] = 1
    # c22 = m1 - m2 + m3 + m6  -> k=1, i=1 -> index 1*2+1=3
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W

def _try_kronecker_decomposition(n, m, p):
    """
    Try to decompose M_{n,m,p} using Kronecker product structure.
    For 3x4x8: treat as 2x2 (Strassen, rank 7) tensor 2x4 (schoolbook, rank 8) = rank 56.
    Actually, let's think more carefully.
    
    3x4x8: n=3, m=4, p=8
    We can write p=8 as 4*2, so the output dimension p*n = 8*3 = 24.
    
    Alternative: write m=4 as 2*2, so input A is 3x4 = 3x(2*2).
    And p=8 as 4*2, so B is 4x8 = (2*2)x(4*2).
    
    Let's try: n=3, m=4=2*2, p=8=4*2.
    A is 3x4, B is 4x8, C is 3x8.
    
    We can view this as:
    - Split m=4 into two blocks of 2: m = m1 * m2 where m1=2, m2=2
    - Split p=8 into two blocks: p = p1 * p2 where p1=4, p2=2
    
    Actually, a cleaner approach:
    3x4x8 can be seen as a block structure.
    Let's use the fact that 4 = 2*2 and 8 = 4*2.
    
    Think of A as 3x(2*2), B as (2*2)x(4*2), C as 3x(4*2).
    
    We can use a Kronecker product of:
    - 2x2x2 multiplication (Strassen, rank 7) for the "inner" 2x2 blocks
    - 3x2x4 multiplication for the "outer" part
    
    Wait, let me reconsider. The standard Kronecker product approach:
    If M_{n1,m1,p1} has rank r1 and M_{n2,m2,p2} has rank r2,
    then M_{n1*n2, m1*m2, p1*p2} has rank at most r1*r2.
    
    So for 3x4x8:
    Option 1: 3x4x8 = (3x2x4) tensor (1x2x2) -> rank(M_{3,2,4}) * rank(M_{1,2,2})
    Option 2: 3x4x8 = (3x2x2) tensor (1x2x4) -> rank(M_{3,2,2}) * rank(M_{1,2,4})
    Option 3: 3x4x8 = (1x4x8) tensor (3x1x1) -> not helpful
    Option 4: 3x4x8 = (3x4x4) tensor (1x1x2) -> rank(M_{3,4,4}) * 1 = rank(M_{3,4,4})
    
    Let's try Option 1: 3x4x8 = (3x2x4) tensor (1x2x2)
    rank(M_{1,2,2}) = 2 (schoolbook, since 1*2*2=4 but actually for 1x2x2, it's just 2 multiplications)
    Wait, M_{1,2,2}: multiply 1x2 by 2x2 to get 1x2. This requires 4 multiplications schoolbook, but actually it's just 2 scalar multiplications for each output, so rank 2? No.
    
    Actually M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    This is just 4 scalar multiplications. Rank is 4 schoolbook.
    
    Hmm, that doesn't help much.
    
    Let me try a different decomposition.
    3x4x8:
    Write 4 = 2*2 and 8 = 2*4.
    So M_{3,4,8} can be seen as a Kronecker product of M_{3,2,2} and M_{1,2,4}.
    rank(M_{3,2,2}) * rank(M_{1,2,4})
    
    M_{3,2,2}: 3x2 times 2x2 = 3x2. Schoolbook rank = 3*2*2 = 12.
    M_{1,2,4}: 1x2 times 2x4 = 1x4. Schoolbook rank = 1*2*4 = 8.
    Total: 12*8 = 96. Worse than schoolbook 96.
    
    Let me try: 3x4x8 = M_{3,4,4} tensor M_{1,1,2}
    rank(M_{3,4,4}) * rank(M_{1,1,2})
    M_{1,1,2}: 1x1 times 1x2 = 1x2. Rank = 2 (just two multiplications).
    So total = rank(M_{3,4,4}) * 2.
    Schoolbook rank(M_{3,4,4}) = 3*4*4 = 48. Total = 96. Same as schoolbook.
    
    What about using Strassen?
    M_{2,2,2} has rank 7 (Strassen).
    
    3x4x8:
    Write as M_{3,2,4} tensor M_{1,2,2}:
    M_{3,2,4}: 3x2 times 2x4 = 3x4. Schoolbook = 24.
    M_{1,2,2}: 1x2 times 2x2 = 1x2. Schoolbook = 4.
    Total = 24*4 = 96.
    
    Let me try: 4x8 = (2*2)x(4*2)
    M_{3,4,8} where 4=2*2, 8=4*2.
    
    Actually, let's use: n=3, m=4, p=8.
    Write m=2*2, p=4*2.
    Then M_{3, 2*2, 4*2} = M_{3,2,4} tensor M_{1,2,2} in some sense.
    
    Hmm, the Kronecker product for matrix multiplication tensors works as:
    If we have M_{n,m,p}, and we write n=n1*n2, m=m1*m2, p=p1*p2,
    then M_{n,m,p} contains M_{n1,m1,p1} tensor M_{n2,m2,p2} as a sub-tensor,
    and rank(M_{n,m,p}) <= rank(M_{n1,m1,p1}) * rank(M_{n2,m2,p2}).
    
    For 3x4x8:
    3 = 3*1, 4 = 2*2, 8 = 4*2
    So M_{3,4,8} <= M_{3,2,4} * M_{1,2,2}
    rank(M_{3,2,4}) * rank(M_{1,2,2})
    Schoolbook: 24 * 4 = 96
    
    3 = 3*1, 4 = 4*1, 8 = 2*4
    M_{3,4,8} <= M_{3,4,2} * M_{1,1,4}
    rank(M_{3,4,2}) * rank(M_{1,1,4})
    Schoolbook: 24 * 4 = 96
    
    Let me try to use Strassen for 2x2x2.
    3x4x8:
    3=3, 4=2*2, 8=4*2
    M_{3,4,8} <= M_{3,2,4} tensor M_{1,2,2}
    But M_{1,2,2} doesn't benefit from Strassen.
    
    What if: 3=1*3, 4=2*2, 8=2*4
    M_{3,4,8} <= M_{1,2,2} tensor M_{3,2,4}
    Same thing.
    
    I think the Kronecker approach with Strassen gives:
    M_{2,2,2} rank 7.
    M_{3,2,4} rank 24 (schoolbook).
    M_{3,4,8} rank <= 7 * 24 = 168? That's worse.
    
    Wait, I think I'm confusing myself. Let me just implement the schoolbook and also try a simple optimization.
    The schoolbook rank is 3*4*8 = 96.
    
    Let me try to find a better decomposition using numerical optimization for rank < 96.
    """
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try Kronecker product with Strassen
    # 3x4x8: use 2x2x2 (Strassen rank 7) and 3x2x4 (schoolbook rank 24)
    # But this gives 7*24 = 168, which is worse.
    
    # Let's just try numerical optimization for rank < 96
    # Start with rank 90 and see if we can find a solution
    best_result = None
    best_rank = n * m * p  # 96
    
    for target_rank in [90, 92, 94]:
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
            
        rng = np.random.default_rng(42)
        nm, mp, pn = n * m, m * p, p * n
        T = _create_target_tensor(n, m, p)
        
        for restart in range(3):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
                
            U = rng.normal(size=(nm, target_rank))
            V = rng.normal(size=(mp, target_rank))
            W = rng.normal(size=(pn, target_rank))
            
            # Normalize to help optimization
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:nm*target_rank].reshape(nm, target_rank)
                V_flat = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_flat = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                T_rec = _reconstruct(U_flat, V_flat, W_flat)
                return np.sum((T_rec - T) ** 2)
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'maxfun': 100})
                x_opt = res.x
                U_opt = x_opt[:nm*target_rank].reshape(nm, target_rank)
                V_opt = x_opt[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_opt = x_opt[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Try rounding to integers or half-integers
                for round_factor in [1, 2]:
                    U_r = np.round(U_opt * round_factor) / round_factor
                    V_r = np.round(V_opt * round_factor) / round_factor
                    W_r = np.round(W_opt * round_factor) / round_factor
                    if _verify_exact(U_r, V_r, W_r, n, m, p):
                        best_result = (U_r, V_r, W_r, target_rank)
                        best_rank = target_rank
                        break
            except Exception:
                pass
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
