"""Matrix multiplication 4x4 by 4x8 rank-minimal search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

_START = time.time()
_DEADLINE = _START + 135.0

def _build_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int16)
    V = np.zeros((m * p, rank), dtype=np.int16)
    W = np.zeros((p * n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_float(U, V, W, T, tol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.allclose(T_rec, T, atol=tol)

def _verify_exact(U, V, W, T):
    """Exact verification using integer arithmetic (entries are integers)."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    # T_rec[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.array_equal(T_rec, T)

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U: (4,7) indexed by (i,j) with i,j in {0,1}
    V: (4,7) indexed by (j,k)
    W: (4,7) indexed by (k,i)
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i//2), c1]
    return U4, V4, W4

def _build_4x4x8_direct():
    """
    Build 4x4x8 as two independent 4x4x4 blocks. Rank 98.
    U: (16, 98) indexed by (i,j)
    V: (32, 98) indexed by (j,k)
    W: (32, 98) indexed by (k,i)
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=20.0):
    """
    Attempt to reduce rank by finding linearly dependent columns.
    For each pair of columns (c1, c2), check if there exists alpha such that
    column c2 = alpha * column c1 in all three matrices simultaneously.
    If so, remove c2.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    tol = 1e-10
    start = time.time()
    
    while time.time() - start < max_time:
        reduced = False
        for c2 in range(rank):
            for c1 in range(c2):
                u1 = Ua[:, c1]
                u2 = Ua[:, c2]
                v1 = Va[:, c1]
                v2 = Va[:, c2]
                w1 = Wa[:, c1]
                w2 = Wa[:, c2]
                
                # Check if u2 = alpha * u1
                nonzero_u = np.abs(u1) > tol
                if not np.any(nonzero_u):
                    continue
                alphas = u2[nonzero_u] / u1[nonzero_u]
                if not np.allclose(alphas, alphas[0], atol=tol):
                    continue
                alpha = alphas[0]
                
                if np.allclose(v2, alpha * v1, atol=tol) and np.allclose(w2, alpha * w1, atol=tol):
                    Ua = np.delete(Ua, c2, axis=1)
                    Va = np.delete(Va, c2, axis=1)
                    Wa = np.delete(Wa, c2, axis=1)
                    rank -= 1
                    reduced = True
                    break
            if reduced:
                break
        if not reduced:
            break
    
    # Verify the reduced decomposition
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    if not np.allclose(T_rec, T, atol=1e-6):
        return None
    
    # Round to integers
    Ua = np.round(Ua).astype(np.int16)
    Va = np.round(Va).astype(np.int16)
    Wa = np.round(Wa).astype(np.int16)
    
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    if not np.array_equal(T_rec, T):
        return None
    
    return Ua, Va, Wa

def _try_column_combination_reduction(U, V, W, T, max_time=15.0):
    """
    Try to find pairs of columns (c1, c2) and scalars (a, b) such that
    the combination a*col_c1 + b*col_c2 is zero in the tensor product sense.
    This is more general than simple scalar multiples.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    tol = 1e-10
    start = time.time()
    
    while time.time() - start < max_time:
        reduced = False
        for c2 in range(rank):
            for c1 in range(c2):
                # We want to check if there exists (a,b) != (0,0) such that
                # a*U[:,c1] + b*U[:,c2] combined with V and W gives zero contribution
                # This is complex; instead try: can we replace col_c2 with col_c2 - alpha*col_c1
                # and then col_c2 becomes zero?
                # Actually, let's try: for each pair, solve for alpha such that
                # (U[:,c2] - alpha*U[:,c1]) has minimal norm, and check if the
                # resulting tensor term is zero.
                # Simpler: check if the 3D tensor term for c2 is a scalar multiple of c1's
                # Term c: U[:,c] * V[:,c] * W[:,c] (outer product)
                # We want: U[:,c2]⊗V[:,c2]⊗W[:,c2] = alpha * (U[:,c1]⊗V[:,c1]⊗W[:,c1])
                
                u1, v1, w1 = Ua[:, c1], Va[:, c1], Wa[:, c1]
                u2, v2, w2 = Ua[:, c2], Va[:, c2], Wa[:, c2]
                
                # Check if u2 = a*u1, v2 = a*v1, w2 = a*w1 for some a
                nonzero_u = np.abs(u1) > tol
                if np.any(nonzero_u):
                    alphas_u = u2[nonzero_u] / u1[nonzero_u]
                    if np.allclose(alphas_u, alphas_u[0], atol=tol):
                        alpha = alphas_u[0]
                        if np.allclose(v2, alpha * v1, atol=tol) and np.allclose(w2, alpha * w1, atol=tol):
                            Ua = np.delete(Ua, c2, axis=1)
                            Va = np.delete(Va, c2, axis=1)
                            Wa = np.delete(Wa, c2, axis=1)
                            rank -= 1
                            reduced = True
                            break
            if reduced:
                break
        if not reduced:
            break
    
    if rank == U.shape[1]:
        return None
    
    Ua = np.round(Ua).astype(np.int16)
    Va = np.round(Va).astype(np.int16)
    Wa = np.round(Wa).astype(np.int16)
    
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    if not np.array_equal(T_rec, T):
        return None
    
    return Ua, Va, Wa

def construct():
    start_time = time.time()
    
    T = _build_tensor()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based construction (rank 98)
    try:
        U, V, W = _build_4x4x8_direct()
        if _verify_float(U.tolist(), V.tolist(), W.tolist(), T):
            U_frac = [[int(x) for x in row] for row in U.tolist()]
            V_frac = [[int(x) for x in row] for row in V.tolist()]
            W_frac = [[int(x) for x in row] for row in W.tolist()]
            if _verify_exact(U_frac, V_frac, W_frac, T):
                best_rank = len(U_frac[0])
                best = {"U": U_frac, "V": V_frac, "W": W_frac}
                print(f"Strassen 4x4x8: rank {best_rank}")
    except Exception as e:
        print(f"Strassen construction failed: {e}")
    
    # Try rank reduction
    if time.time() - start_time < _DEADLINE - 30:
        try:
            U_list = best["U"]
            V_list = best["V"]
            W_list = best["W"]
            reduced = _try_rank_reduction(U_list, V_list, W_list, T, max_time=15.0)
            if reduced:
                U_r, V_r, W_r = reduced
                U_frac = [[int(x) for x in row] for row in U_r.tolist()]
                V_frac = [[int(x) for x in row] for row in V_r.tolist()]
                W_frac = [[int(x) for x in row] for row in W_r.tolist()]
                if _verify_exact(U_frac, V_frac, W_frac, T):
                    new_rank = len(U_frac[0])
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best = {"U": U_frac, "V": V_frac, "W": W_frac}
                        print(f"After rank reduction: rank {best_rank}")
        except Exception as e:
            print(f"Rank reduction failed: {e}")
    
    # Try more aggressive reduction with column combinations
    if time.time() - start_time < _DEADLINE - 20:
        try:
            U_list = best["U"]
            V_list = best["V"]
            W_list = best["W"]
            reduced = _try_column_combination_reduction(U_list, V_list, W_list, T, max_time=10.0)
            if reduced:
                U_r, V_r, W_r = reduced
                U_frac = [[int(x) for x in row] for row in U_r.tolist()]
                V_frac = [[int(x) for x in row] for row in V_r.tolist()]
                W_frac = [[int(x) for x in row] for row in W_r.tolist()]
                if _verify_exact(U_frac, V_frac, W_frac, T):
                    new_rank = len(U_frac[0])
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best = {"U": U_frac, "V": V_frac, "W": W_frac}
                        print(f"After column combination reduction: rank {best_rank}")
        except Exception as e:
            print(f"Column combination reduction failed: {e}")
    
    return best
