import numpy as np
import time
import scipy.optimize as opt
from scipy.linalg import lstsq as spl_lstsq
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try rank-16 via CW embedding
    result_16 = _try_rank16_cw(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    result_17 = _rank17_decomposition(n, m, p)
    return result_17

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact (half-integer entries)."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _winograd_f1x3x3():
    """
    Winograd F-algorithm for 1x3x3 matrix multiplication.
    
    The 1x3x3 tensor T[i,j,k] = delta(i,0) is the identity in i (trivial since n=1).
    Actually for 1x3x3: T[i,j,k] = 1 if i=0 (only index), so T[0,j,k] = 1 for all j,k.
    Wait - the 1x3x3 tensor is: output[i,j] = sum_k A[i,k]*B[k,j] where i in {0}, j in {0,1,2}, k in {0,1,2}
    So T[i,j,k] = delta(i,0) * 1 = 1 for all j,k (since i is always 0).
    
    Actually the tensor for 1x3x3 matmul is T[0,j,k] = 1 for all j,k.
    This is just the outer product of ones: T = 1_i * 1_j * 1_k... no.
    
    For matrix multiplication C = A*B where A is 1x3, B is 3x3, C is 1x3:
    C[j] = sum_k A[k] * B[k,j]
    So T[k, j, l] = 1 where k indexes A's column (=B's row), j indexes B's column, l is A's row (=C's row, always 0).
    
    In our convention: U[i,j,r] for A (n x m), V[j,k,r] for B (m x p), W[k,i,r] for C output.
    For 1x3x3: n=1, m=3, p=3. T[i,j,k] where i in [0,1), j in [0,3), k in [0,3).
    T[0,j,k] = 1 for all j,k.
    
    Winograd F_2 for 1x3: We can use the F_2 algorithm structure.
    For 1x3x3, the minimal rank is known to be 7 (from 3x3x3 Strassen applied with one dimension=1).
    
    Actually, let's think more carefully. The 1x3x3 tensor has rank at most 3 (just use 3 scalar multiplications: C[j] = A[0,j]*B[j,j]... no that's not right).
    
    C[j] = sum_{k=0}^{2} A[k] * B[k,j] for j=0,1,2.
    This is 9 scalar multiplications in schoolbook. But we can do better.
    
    Actually for 1x3x3, the tensor is T[k,j,l] = delta(l,0) for the "A" index k, "B" indices (k,j), "C" index j.
    Wait I'm confusing myself. Let me use the standard convention.
    
    For multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,j] = sum_k A[i,k] * B[k,j]
    Tensor: T[i,k,j] = 1 (all ones)
    
    For 1x3x3: n=1, m=3, p=3
    T[i,k,j] = 1 for i in {0}, k in {0,1,2}, j in {0,1,2}
    So T[0,k,j] = 1 for all k,j.
    
    This tensor is just the 3x3 all-ones matrix (flattened). Its rank as a 3x3x1 tensor... 
    Actually it's a 1x3x3 tensor. The rank of the all-ones 1x3x3 tensor is 1! Because T[0,k,j] = 1*1*1 = u[0,k]*v[k,j]*w[j,0] where u[0,k]=1, v[k,j]=1, w[j,0]=1.
    
    Wait, that's not right either. The tensor T[i,k,j] for matmul is always all-ones. For n=1,m=3,p=3, T[0,k,j]=1 for all k,j.
    Rank-1: T[0,k,j] = U[0,k,r]*V[k,j,r]*W[j,0,r]. If U[0,k]=1, V[k,j]=1, W[j,0]=1, then product = 1. So rank 1!
    
    But that can't be right because 1x3x3 matmul needs at least 3 multiplications...
    
    OH WAIT. I think I'm confusing the tensor structure. Let me re-read the problem.
    
    "Matrix multiplication 2,3,3" - multiply 2x3 by 3x3.
    C[i,j] = sum_k A[i,k] * B[k,j] where i=0,1; k=0,1,2; j=0,1,2.
    
    The tensor is T[i,k,j] = 1 for all i,k,j. Shape: (2,3,3).
    
    For the sub-problem 1x3x3: T[i,k,j] = 1 for i=0, k=0,1,2, j=0,1,2. Shape: (1,3,3).
    
    In the factorization: sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] = T[i,k,j] = 1.
    
    For the 1x3x3 sub-tensor: sum_r U[0,k,r] * V[k,j,r] * W[j,0,r] = 1 for all k,j.
    
    This is equivalent to: the matrix M_r[k,j] = U[0,k,r]*V[k,j,r] and N_r[j] = W[j,0,r], and we need sum_r M_r[k,j]*N_r[j] = 1.
    
    Hmm, this is a bilinear form. The rank of this tensor... Let me think of it differently.
    
    Actually, T[i,k,j]=1 for all indices means the tensor is the all-ones tensor. The tensor rank of a 1x3x3 all-ones tensor is 1 (it's an outer product of three all-ones vectors).
    
    So the 1x3x3 sub-problem has rank 1! That means 2x3x3 can be decomposed as:
    - 2 terms for the 2x1x1 part (the "i" dimension interaction)
    - 1 term for the 1x3x3 part
    Total: 3? That can't be right either since we know rank >= 15.
    
    I think I'm fundamentally misunderstanding the tensor structure. Let me re-read.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    where i in [0,n), j in [0,m), k in [0,p).
    
    For n=2, m=3, p=3:
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    
    This is the standard matrix multiplication tensor. The "1x3x3 sub-tensor" doesn't make sense in isolation because the tensor is entangled across all three dimensions.
    
    Let me reconsider the CW embedding approach. The Coppersmith-Winograd method for small matrices typically involves:
    1. Embedding the problem into a larger problem
    2. Using the F-algorithm (Winograd) to get a structured decomposition
    3. Solving for the coefficients
    
    For 2x3x3, one approach is:
    - View it as a block matrix multiplication
    - Use the structure to reduce the number of terms
    
    Let me try a different approach: directly set up the rank-16 problem as a system of equations and solve it.
    
    The constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k) in [0,2)x[0,3)x[0,3).
    That's 18 equations.
    
    With rank 16, we have 16*6 + 16*9 + 16*6 = 96 + 144 + 96 = 336 variables and 18 equations.
    This is very underdetermined, so solutions should exist if rank 16 is achievable.
    
    The challenge is finding solutions with half-integer entries.
    
    Let me try a structured approach: use the known rank-17 solution and try to merge two terms.
    """
    pass

def _try_rank16_cw(n, m, p, deadline):
    """
    Try to find rank-16 decomposition using structured ansatz.
    
    Approach: Start from the rank-17 decomposition and try to merge
    two terms by finding a rank-1 correction. Alternatively, use a
    structured parameterization inspired by Winograd's F-algorithm.
    
    The key insight: for 2x3x3, we can write the tensor as:
    T[i,j,k] = T_2x1x1[i,j,k] + T_1x3x3[i,j,k] + cross terms
    
    where the 2x1x1 part handles the first column (j=0) and the
    1x3x3 part handles the remaining structure.
    
    We'll try multiple structured initializations and polish them.
    """
    rank = 16
    
    # Get the rank-17 solution as a starting point
    sol17 = _rank17_decomposition(n, m, p)
    U17 = np.array(sol17["U"])
    V17 = np.array(sol17["V"])
    W17 = np.array(sol17["W"])
    
    # Strategy 1: Try to merge pairs of terms from rank-17 into rank-16
    # by finding a rank-1 "correction" that absorbs two terms
    for pair in range(17):
        if time.time() > deadline:
            break
        for pair2 in range(pair+1, 17):
            if time.time() > deadline:
                break
            
            # Remove terms pair and pair2, try to replace with 1 term
            U_reduced = np.delete(U17, [pair, pair2], axis=1)
            V_reduced = np.delete(V17, [pair, pair2], axis=1)
            W_reduced = np.delete(W17, [pair, pair2], axis=1)
            
            # The residual that the new term must fill
            U_t = U_reduced.reshape(n, m, rank-2)
            V_t = V_reduced.reshape(m, p, rank-2)
            W_t = W_reduced.reshape(p, n, rank-2)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            residual = 1.0 - prod  # What the new term must contribute
            
            # The new term must satisfy: U_new[i,j] * V_new[j,k] * W_new[k,i] = residual[i,j,k]
            # This is a rank-1 tensor equation. Check if residual has rank 1.
            # Reshape residual as a 2x3x3 tensor and check its tensor rank.
            # For rank-1: residual[i,j,k] = a[i,j] * b[j,k] * c[k,i]
            
            # Try to find a,b,c such that a[i,j]*b[j,k]*c[k,i] = residual[i,j,k]
            # Fix c[k,i] = 1 for all (k,i) (gauge freedom), then:
            # a[i,j]*b[j,k] = residual[i,j,k] for all i,j,k
            # This means: for fixed i,k: a[i,j]*b[j,k] = residual[i,j,k]
            # Sum over j: this gives a system.
            
            # Alternative: use the fact that residual must factorize.
            # Check: residual[i,j,k] * residual[i',j',k'] = residual[i,j',k'] * residual[i',j,k]
            # (for appropriate index combinations)
            
            # Simpler: try to solve for a, b, c directly
            # a has 6 entries, b has 9 entries, c has 6 entries = 21 unknowns
            # 18 equations (one per (i,j,k))
            
            # Initialize with random values and use iterative method
            np.random.seed(pair * 100 + pair2)
            a = np.random.randn(n, m) * 0.5 + 0.5  # Keep positive for stability
            b = np.random.randn(m, p) * 0.5 + 0.5
            c = np.random.randn(p, n) * 0.5 + 0.5
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Current product
                cur = np.einsum('ij,jk,ki->ijk', a, b, c)
                err = np.sum((cur - residual)**2)
                
                if err < 1e-12:
                    # Found a valid merge!
                    U_new = np.zeros((n*m, rank-1))
                    V_new = np.zeros((m*p, rank-1))
                    W_new = np.zeros((p*n, rank-1))
                    
                    U_new[:, :rank-2] = U_reduced
                    V_new[:, :rank-2] = V_reduced
                    W_new[:, :rank-2] = W_reduced
                    
                    U_new[:, rank-2] = a.reshape(n*m)
                    V_new[:, rank-2] = b.reshape(m*p)
                    W_new[:, rank-2] = c.reshape(p*n)
                    
                    ok, Us, Vs, Ws = _check_exact(U_new, V_new, W_new, n, m, p)
                    if ok:
                        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    
                    # Even if not exact half-integer, try polishing
                    Us, Vs, Ws = _polish_rank16(U_new, V_new, W_new, n, m, p, deadline)
                    if Us is not None:
                        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    break
                
                # Gradient update
                # d(err)/da[i,j] = 2 * sum_{k} (cur[i,j,k] - residual[i,j,k]) * b[j,k] * c[k,i]
                diff = cur - residual
                da = 2 * np.einsum('ijk,jk,ki->ij', diff, b, c)
                db = 2 * np.einsum('ijk,ij,ki->jk', diff, a, c)
                dc = 2 * np.einsum('ijk,ij,jk->ki', diff, a, b)
                
                lr = 0.1 / (1 + it * 0.01)
                a -= lr * da
                b -= lr * db
                c -= lr * dc
            
            # Try with different gauge: fix a instead
            if time.time() > deadline:
                break
            np.random.seed(pair * 200 + pair2 + 7)
            a = np.ones((n, m))
            b = np.random.randn(m, p) * 0.5 + 0.5
            c = np.random.randn(p, n) * 0.5 + 0.5
            
            for it in range(200):
                if time.time() > deadline:
                    break
                
                cur = np.einsum('ij,jk,ki->ijk', a, b, c)
                err = np.sum((cur - residual)**2)
                
                if err < 1e-12:
                    U_new = np.zeros((n*m, rank-1))
                    V_new = np.zeros((m*p, rank-1))
                    W_new = np.zeros((p*n, rank-1))
                    
                    U_new[:, :rank-2] = U_reduced
                    V_new[:, :rank-2] = V_reduced
                    W_new[:, :rank-2] = W_reduced
                    
                    U_new[:, rank-2] = a.reshape(n*m)
                    V_new[:, rank-2] = b.reshape(m*p)
                    W_new[:, rank-2] = c.reshape(p*n)
                    
                    ok, Us, Vs, Ws = _check_exact(U_new, V_new, W_new, n, m, p)
                    if ok:
                        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    
                    Us, Vs, Ws = _polish_rank16(U_new, V_new, W_new, n, m, p, deadline)
                    if Us is not None:
                        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    break
                
                diff = cur - residual
                db = 2 * np.einsum('ijk,ij,ki->jk', diff, a, c)
                dc = 2 * np.einsum('ijk,ij,jk->ki', diff, a, b)
                
                lr = 0.1 / (1 + it * 0.01)
                b -= lr * db
                c -= lr * dc
    
    # Strategy 2: Direct structured rank-16 search
    # Use a parameterization where the 16 terms have specific structure
    for seed in range(50):
        if time.time() > deadline:
            break
        
        U, V, W = _structured_init16(seed, n, m, p)
        
        # Gradient descent
        lr = 0.05
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(400):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            if step % 50 == 0 and step > 0:
                lr *= 0.93
            
            U -= lr * dU.reshape(n*m, rank)
            V -= lr * dV.reshape(m*p, rank)
            W -= lr * dW.reshape(p*n, rank)
            
            if step % 25 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
        
        if best_err < 1e-4:
            U_b, V_b, W_b = best_state
            Us, Vs, Ws = _polish_rank16(U_b, V_b, W_b, n, m, p, deadline)
            if Us is not None:
                return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
        
        if best_err < 1e-6:
            Us, Vs, Ws = _polish_rank16(U, V, W, n, m, p, deadline)
            if Us is not None:
                return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    return None

def _structured_init16(seed, n, m, p):
    """Structured initialization for rank-16 search."""
    np.random.seed(seed)
    rank = 16
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Base structure: use the rank-17 pattern but with one term zeroed
    # and perturbations on others
    sol17 = _rank17_decomposition(n, m, p)
    U17 = np.array(sol17["U"])
    V17 = np.array(sol17["V"])
    W17 = np.array(sol17["W"])
    
    # Drop a random term and perturb the rest
    drop = seed % 17
    U[:, :rank] = np.delete(U17, drop, axis=1)
    V[:, :rank] = np.delete(V17, drop, axis=1)
    W[:, :rank] = np.delete(W17, drop, axis=1)
    
    # Add small perturbations to all entries
    U += np.random.randn(n*m, rank) * 0.05
    V += np.random.randn(m*p, rank) * 0.05
    W += np.random.randn(p*n, rank) * 0.05
    
    return U, V, W

def _polish_rank16(U, V, W, n, m, p, deadline, max_iters=100):
    """High-precision polish for rank-16."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        err = np.sum(res**2)
        
        if err < 1e-14:
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return Us, Vs, Ws
        
        dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-15:
            break
        lr = min(0.5, 1e-6 / grad_norm)
        
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        if err < 1e-10:
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return Us, Vs, Ws
    
    if err < 1e-8:
        ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
        if ok:
            return Us, Vs, Ws
    
    return None, None, None
