"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Try to find a rank-10 decomposition for the 2x3x2 sub-tensor
    best = None
    best_rank = n * m * p  # schoolbook 48

    # First, try the known rank-44 solution (Strassen 7 + 4 per 2x3x2 block)
    if time.time() < deadline - 30:
        try:
            U44, V44, W44 = _build_rank44(n, m, p)
            if _verify(n, m, p, U44, V44, W44):
                best = (U44, V44, W44)
                best_rank = 44
        except Exception:
            pass

    # Try to find rank-10 for 2x3x2 sub-tensor via structured search
    if time.time() < deadline - 15:
        try:
            sub_U, sub_V, sub_W = _find_rank10_2x3x2()
            if sub_U is not None:
                U10, V10, W10 = _build_from_subtensor(n, m, p, sub_U, sub_V, sub_W)
                if _verify(n, m, p, U10, V10, W10):
                    best = (U10, V10, W10)
                    best_rank = 40
        except Exception:
            pass

    # Try rank-43 via perturbation of the rank-44 solution
    if best is not None and best_rank == 44 and time.time() < deadline - 10:
        try:
            U43, V43, W43 = _try_rank43(n, m, p, best[0], best[1], best[2])
            if U43 is not None and _verify(n, m, p, U43, V43, W43):
                best = (U43, V43, W43)
                best_rank = 43
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Fallback: schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _build_rank44(n, m, p):
    """Build the rank-44 solution: 4 blocks of 2x3x2, each with Strassen(7)+4."""
    block_size = p // 4  # 2
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(4):
        k_offset = b * block_size
        col_offset = b * rank_per_block

        # Strassen 2x2*2x2 for A[0:2, 0:2] * B[0:2, k_offset:k_offset+2]
        # A indices: i in {0,1}, j in {0,1} -> U[i*m+j, col]
        # B indices: j in {0,1}, k in {k_offset, k_offset+1} -> V[j*p+k, col]
        # C indices: i in {0,1}, k in {k_offset, k_offset+1} -> W[k*n+i, col]

        # Strassen multipliers:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # U contributions (A side)
        strassen_U = [
            {0: 1, 4: 1},      # M1: A00+A11
            {3: 1, 4: 1},      # M2: A10+A11
            {0: 1},            # M3: A00
            {4: 1},            # M4: A11
            {0: 1, 1: 1},      # M5: A00+A01
            {3: 1, 0: -1},     # M6: A10-A00
            {1: 1, 4: -1},     # M7: A01-A11
        ]

        # V contributions (B side), B[j, k_offset+kk] -> V[j*p + k_offset+kk, col]
        strassen_V = [
            {0*p + k_offset: 1, 1*p + k_offset+1: 1},    # M1: B00+B11
            {0*p + k_offset: 1},                          # M2: B00
            {1*p + k_offset: 1, 1*p + k_offset+1: -1},    # M3: B10-B11
            {0*p + k_offset+1: 1, 0*p + k_offset: -1},    # M4: B01-B00
            {1*p + k_offset+1: 1},                        # M5: B11
            {0*p + k_offset: 1, 0*p + k_offset+1: 1},     # M6: B00+B01
            {1*p + k_offset: 1, 1*p + k_offset+1: 1},     # M7: B10+B11
        ]

        # W contributions (C side), C[i, k_offset+kk] -> W[(k_offset+kk)*n + i, col]
        strassen_W = [
            {(k_offset)*n + 0: 1, (k_offset+1)*n + 1: 1},   # M1: C00+=1, C11+=1
            {(k_offset)*n + 1: 1},                           # M2: C10+=1
            {(k_offset+1)*n + 0: 1, (k_offset+1)*n + 1: 1}, # M3: C01+=1, C11+=1
            {(k_offset)*n + 0: 1, (k_offset)*n + 1: 1},     # M4: C00+=1, C10+=1
            {(k_offset)*n + 0: -1, (k_offset+1)*n + 0: 1},  # M5: C00-=1, C01+=1
            {(k_offset+1)*n + 1: 1},                         # M6: C11+=1
            {(k_offset)*n + 0: 1},                           # M7: C00+=1
        ]

        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val

        # Outer product: A[0:2, 2] * B[2, k_offset:k_offset+2]
        # 4 terms: for i in {0,1}, kk in {0,1}
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W


def _find_rank10_2x3x2():
    """
    Try to find a rank-10 decomposition for the 2x3x2 tensor.
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}.
    
    We seek: 1 = sum_{c=0}^{9} U[i,j,c] * V[j',k,c] * W[k',i,c]
    where for each c, U is nonzero at one (i,j), V at one (j',k), W at one (k',i),
    and j=j', k=k', i=i' for the product to be nonzero.
    
    Actually, the correct form is:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[l*p+k, c] B[l,k])
    
    For the sub-tensor (2x3x2), we need:
    For each (i,j,k) in {0,1}x{0,1,2}x{0,1}:
      1 = sum_c U_sub[i,j,c] * V_sub[j,k,c] * W_sub[k,i,c]
    
    where U_sub is 2x3x10, V_sub is 3x2x10, W_sub is 2x2x10.
    
    This is a system of 12 polynomial equations in 180 unknowns (10*6 + 10*6 + 10*4 = 60+60+40=160... wait)
    U_sub: 2*3*10 = 60
    V_sub: 3*2*10 = 60
    W_sub: 2*2*10 = 40
    Total: 160 unknowns, 12 equations.
    
    This is highly underdetermined, so solutions should exist. Let me try a structured approach.
    """
    n, m, p = 2, 3, 2
    rank = 10
    
    # Initialize with the rank-11 solution and try to nullify one term
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    
    # Start with Strassen(7) + 4 trivial = 11 terms, then try to merge one
    # Actually, let me try a different approach: use 10 terms with a mix
    
    # Let me try to construct it directly.
    # The 2x3x2 tensor can be seen as:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    
    # We need 10 multiplications to compute these 4 outputs (each is a sum of 3 products).
    
    # Let me try: use 6 for the 2x2 core (A[0:2,0:2] * B[0:2,0:2]) and 4 for the rest.
    # But 2x2*2x2 needs 7 (Strassen) or 4 (schoolbook).
    
    # Alternative: use a non-Strassen approach.
    # Let me try to find a solution via random initialization and Newton's method.
    
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n, m, rank)
    V = np.random.randn(m, p, rank)
    W = np.random.randn(p, n, rank)
    
    # Define the residual function
    def residuals(U, V, W):
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    res.append(val - 1.0)
        return np.array(res)
    
    # Try Newton's method
    for iteration in range(100):
        r = residuals(U, V, W)
        if np.max(np.abs(r)) < 1e-10:
            break
        
        # Compute Jacobian numerically
        jac = np.zeros((12, 160))
        eps = 1e-7
        idx = 0
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    U_orig = U[i, j, c]
                    U[i, j, c] = U_orig + eps
                    r_plus = residuals(U, V, W)
                    U[i, j, c] = U_orig - eps
                    r_minus = residuals(U, V, W)
                    U[i, j, c] = U_orig
                    jac[:, idx] = (r_plus - r_minus) / (2 * eps)
                    idx += 1
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    V_orig = V[j, k, c]
                    V[j, k, c] = V_orig + eps
                    r_plus = residuals(U, V, W)
                    V[j, k, c] = V_orig - eps
                    r_minus = residuals(U, V, W)
                    V[j, k, c] = V_orig
                    jac[:, idx] = (r_plus - r_minus) / (2 * eps)
                    idx += 1
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    W_orig = W[k, i, c]
                    W[k, i, c] = W_orig + eps
                    r_plus = residuals(U, V, W)
                    W[k, i, c] = W_orig - eps
                    r_minus = residuals(U, V, W)
                    W[k, i, c] = W_orig
                    jac[:, idx] = (r_plus - r_minus) / (2 * eps)
                    idx += 1
        
        # Solve least squares
        try:
            delta, _, _, _ = np.linalg.lstsq(jac, -r, rcond=None)
        except:
            break
        
        # Update
        idx = 0
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    U[i, j, c] += delta[idx]
                    idx += 1
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    V[j, k, c] += delta[idx]
                    idx += 1
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    W[k, i, c] += delta[idx]
                    idx += 1
    
    r = residuals(U, V, W)
    if np.max(np.abs(r)) < 1e-8:
        # Check if entries are close to integers/half-integers
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        rounded = np.round(all_vals * 2) / 2
        if np.max(np.abs(all_vals - rounded)) < 1e-6:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            return U, V, W
    
    return None, None, None


def _build_from_subtensor(n, m, p, sub_U, sub_V, sub_W):
    """Build the full 2x3x8 decomposition from a 2x3x2 sub-tensor decomposition."""
    sub_rank = sub_U.shape[2]
    num_blocks = p // 2  # 4 blocks
    total_rank = sub_rank * num_blocks
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(num_blocks):
        k_offset = b * 2
        col_offset = b * sub_rank
        
        for i in range(n):
            for j in range(m):
                for c in range(sub_rank):
                    U[i * m + j, col_offset + c] = sub_U[i, j, c]
        
        for j in range(m):
            for k in range(2):
                for c in range(sub_rank):
                    V[j * p + k_offset + k, col_offset + c] = sub_V[j, k, c]
        
        for k in range(2):
            for i in range(n):
                for c in range(sub_rank):
                    W[(k_offset + k) * n + i, col_offset + c] = sub_W[k, i, c]
    
    return U, V, W


def _try_rank43(n, m, p, U44, V44, W44):
    """Try to reduce rank-44 to rank-43 by finding a linear dependency."""
    # This is a placeholder; a proper implementation would use
    # algebraic geometry techniques to find if rank 43 is achievable.
    return None, None, None


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact.
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(U.shape[1]):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j, c] * A[i, j]
                v_val = 0.0
                for l in range(m):
                    v_val += V[l * p + k, c] * B[l, k]
                C_comp[i, k] += W[k * n + i, c] * u_val * v_val
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
