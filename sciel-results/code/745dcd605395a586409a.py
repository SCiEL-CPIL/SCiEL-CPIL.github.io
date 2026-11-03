import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safe margin before 150s kill

    n, m, p = 6, 6, 6

    def time_left():
        return deadline - time.time()

    # Trivial 6x6 decomposition (rank 216) as guaranteed fallback
    def trivial_6x6():
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i*m+k, col] = 1
                    V[k*p+j, col] = 1
                    W[j*n+i, col] = 1
                    col += 1
        return U, V, W

    # Strassen 2x2 block (3x3 subblocks) decomposition: rank 7*27 = 189
    def strassen_3x3_blocks():
        bs = 3
        nb = 2
        strassen_2x2 = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        trivial_3x3 = [(i,k,j) for i in range(3) for k in range(3) for j in range(3)]
        R = 7 * 27
        U = np.zeros((n*m, R), dtype=np.int64)
        V = np.zeros((m*p, R), dtype=np.int64)
        W = np.zeros((p*n, R), dtype=np.int64)
        col = 0
        for (sA, sB, sC) in strassen_2x2:
            for (ti, tk, tj) in trivial_3x3:
                for (sbr, sbc, sc) in sA:
                    U[(sbr*bs + ti)*m + (sbc*bs + tk), col] += sc
                for (sbr, sbc, sc) in sB:
                    V[(sbr*bs + tk)*p + (sbc*bs + tj), col] += sc
                for (sbr, sbc, sc) in sC:
                    W[(sbr*bs + tj)*n + (sbc*bs + ti), col] += sc
                col += 1
        return U, V, W

    # Fast verification: check all 216 entries of the matrix multiplication tensor
    def verify(Um, Vm, Wm):
        r = Um.shape[1]
        # T[i,j,k] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
        # We need T[i,j,k] = 1 for all i,j,k
        U_r = Um.reshape(n, m, r)
        V_r = Vm.reshape(m, p, r)
        W_r = Wm.reshape(p, n, r)
        # Use einsum for speed
        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    # Try Strassen construction
    U, V, W = strassen_3x3_blocks()
    
    # Verify; if fails, fall back to trivial
    if not verify(U, V, W):
        U, V, W = trivial_6x6()
        if not verify(U, V, W):
            # Should never happen, but return something
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_r = current_U.shape[1]

    # Greedy removal with fast verification
    # Precompute the contribution of each term for faster deficit check
    # For each term t, its contribution to T[i,j,k] is U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    # When removing term t, we need: for all (i,j,k), sum_{r != t} U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1
    # Equivalently: the current sum is 1, and we subtract term t's contribution, so we need
    # current_sum - contribution_t == 1 for all (i,j,k) where contribution_t != 0
    # i.e., contribution_t must be 0 for all (i,j,k) where current_sum != 1, 
    # but current_sum IS 1 everywhere (valid decomposition), so we need contribution_t == 0 everywhere? No.
    # Actually: after removal, sum_{r != t} = 1. Since sum_all = 1, we need contribution_t = 0 everywhere.
    # That's too strict. The issue is that multiple terms contribute to the same (i,j,k).
    # So we just need to verify after removal.
    
    # For speed, batch the verification
    def try_remove_batch(Um, Vm, Wm, candidates):
        """Try removing each candidate; return list of (index, success)"""
        results = []
        for t in candidates:
            if time_left() < 5:
                break
            r = Um.shape[1]
            mask = np.ones(r, dtype=bool)
            mask[t] = False
            Um_c = Um[:, mask]
            Vm_c = Vm[:, mask]
            Wm_c = Wm[:, mask]
            
            if verify(Um_c, Vm_c, Wm_c):
                results.append((t, True))
                # Only return the first success to maintain greedy
                break
            else:
                results.append((t, False))
        return results

    # Multiple passes of greedy removal
    max_passes = 50
    for pass_num in range(max_passes):
        if time_left() < 15:
            break
        removed = False
        # Try from the end (terms added last may be more redundant)
        for t in range(current_r - 1, -1, -1):
            if time_left() < 15:
                break
            r = current_r
            mask = np.ones(r, dtype=bool)
            mask[t] = False
            Um_c = current_U[:, mask]
            Vm_c = current_V[:, mask]
            Wm_c = current_W[:, mask]
            
            if verify(Um_c, Vm_c, Wm_c):
                current_U = Um_c
                current_V = Vm_c
                current_W = Wm_c
                current_r -= 1
                removed = True
                break  # restart scan after removal
        if not removed:
            break

    # Final verification
    if not verify(current_U, current_V, current_W):
        # Fallback: return the initial Strassen or trivial
        U_init, V_init, W_init = strassen_3x3_blocks()
        if verify(U_init, V_init, W_init):
            current_U, current_V, current_W = U_init, V_init, W_init
        else:
            U_init, V_init, W_init = trivial_6x6()
            current_U, current_V, current_W = U_init, V_init, W_init

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
