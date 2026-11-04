import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        U_3d = np.array(U).reshape(n, m, rank)
        V_3d = np.array(V).reshape(m, p, rank)
        W_3d = np.array(W).reshape(p, n, rank)
        S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(S, 1.0, atol=1e-9):
            return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # c=0: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # c=1: M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # c=2: M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 0] = 1; W[3, 1] = 1
        # c=3: M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 0] = 1; W[2, 1] = 1
        # c=4: M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 0] = -1; W[1, 0] = 1
        # c=5: M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 1] = 1
        # c=6: M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 0] = 1
        return U, V, W

    U2, V2, W2 = strassen_2x2_2x2()
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    S2 = np.einsum('ijc,jkc,kic->ijk', U2_3d, V2_3d, W2_3d)
    assert np.allclose(S2, 1.0, atol=1e-9), f"Strassen 2x2x2 failed: {S2}"

    def make_4x4x4_strassen():
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        for c1 in range(7):
            for c0 in range(7):
                c = c1 * 7 + c0
                for i1 in range(2):
                    for j1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                i = 2*i1 + i0
                                j = 2*j1 + j0
                                U[i*4+j, c] = U2[i1*2+j1, c1] * U2[i0*2+j0, c0]
                for j1 in range(2):
                    for k1 in range(2):
                        for j0 in range(2):
                            for k0 in range(2):
                                j = 2*j1 + j0
                                k = 2*k1 + k0
                                V[j*4+k, c] = V2[j1*2+k1, c1] * V2[j0*2+k0, c0]
                for k1 in range(2):
                    for i1 in range(2):
                        for k0 in range(2):
                            for i0 in range(2):
                                k = 2*k1 + k0
                                i = 2*i1 + i0
                                W[k*4+i, c] = W2[k1*2+i1, c1] * W2[k0*2+i0, c0]
        return U, V, W

    U444, V444, W444 = make_4x4x4_strassen()
    U444_3d = U444.reshape(4, 4, 49)
    V444_3d = V444.reshape(4, 4, 49)
    W444_3d = W444.reshape(4, 4, 49)
    S444 = np.einsum('ijc,jkc,kic->ijk', U444_3d, V444_3d, W444_3d)
    assert np.allclose(S444, 1.0, atol=1e-9), f"4x4x4 Strassen failed: {S444}"

    def make_4x4x1_schoolbook():
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        c = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j, c] = 1
                V[j*5+4, c] = 1
                W[4*4+i, c] = 1
                c += 1
        return U, V, W

    U441, V441, W441 = make_4x4x1_schoolbook()

    total_rank = 65
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((20, total_rank))
    W_total = np.zeros((20, total_rank))
    U_total[:, :49] = U444
    V_total[:, :49] = V444
    W_total[:, :49] = W444
    U_total[:, 49:] = U441
    V_total[:, 49:] = V441
    W_total[:, 49:] = W441

    U_total_3d = U_total.reshape(4, 4, 65)
    V_total_3d = V_total.reshape(4, 5, 65)
    W_total_3d = W_total.reshape(5, 4, 65)
    S_total = np.einsum('ijc,jkc,kic->ijk', U_total_3d, V_total_3d, W_total_3d)
    assert np.allclose(S_total, 1.0, atol=1e-9), f"4x4x5 combined failed: {S_total}"

    best = {"U": U_total.copy(), "V": V_total.copy(), "W": W_total.copy()}
    best_rank = total_rank

    def get_error_count(rank, U, V, W):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.sum(np.abs(S - 1.0) > 1e-9)

    def get_smooth_error(rank, U, V, W):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.sum((S - 1.0) ** 2)

    def local_search_smooth(rank, U, V, W, seed, max_steps=300):
        rng = random.Random(seed)
        U_ = U.copy()
        V_ = V.copy()
        W_ = W.copy()
        
        cur_smooth = get_smooth_error(rank, U_, V_, W_)
        cur_err = get_error_count(rank, U_, V_, W_)
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        
        best_smooth = cur_smooth
        best_err = cur_err
        no_improve = 0
        
        candidate_vals = [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if cur_err == 0:
                break
            
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    U_[idx, c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    U_[idx, c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        U_[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    V_[idx, c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    V_[idx, c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        V_[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    W_[idx, c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    W_[idx, c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        W_[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 80:
                break
        
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        return None

    # Try to reduce rank from 65 downward
    for rank in range(64, 30):
        if time.time() > deadline - 10:
            break
        
        num_restarts = 3
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            # Initialize with structured random values
            U0 = np.zeros((n*m, rank))
            V0 = np.zeros((m*p, rank))
            W0 = np.zeros((p*n, rank))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.5:
                            U0[i*m+j, c] = rng.choice([1, -1, 0.5, -0.5])
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.5:
                            V0[j*p+k, c] = rng.choice([1, -1, 0.5, -0.5])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.5:
                            W0[k*n+i, c] = rng.choice([1, -1, 0.5, -0.5])
            
            result = local_search_smooth(rank, U0, V0, W0, seed + 7777, max_steps=100)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break

    if isinstance(best['U'], np.ndarray):
        best['U'] = best['U'].tolist()
        best['V'] = best['V'].tolist()
        best['W'] = best['W'].tolist()

    return best
