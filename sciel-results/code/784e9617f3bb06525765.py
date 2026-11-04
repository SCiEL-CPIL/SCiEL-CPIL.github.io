"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-8):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_decompose_subtensor(target_t, rank, max_iter=100, tol=1e-12, num_restarts=10, time_budget=None):
        """ALS decomposition of a given target tensor with multiple restarts."""
        Nt, Mt, Pt = target_t.shape
        best_err = np.inf
        best_UVW = None
        
        for restart in range(num_restarts):
            if time_budget is not None and time_budget() < 15:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                U = np.random.randn(Nt, rank) * 0.1
                V = np.random.randn(Mt, rank) * 0.1
                W = np.random.randn(Pt, rank) * 0.1
            elif restart % 3 == 1:
                U = np.random.randn(Nt, rank) * 0.01
                V = np.random.randn(Mt, rank) * 0.01
                W = np.random.randn(Pt, rank) * 0.01
            else:
                U = np.random.randn(Nt, rank)
                V = np.random.randn(Mt, rank)
                W = np.random.randn(Pt, rank)
                U /= np.linalg.norm(U) + 1e-10
                V /= np.linalg.norm(V) + 1e-10
                W /= np.linalg.norm(W) + 1e-10
            
            prev_err = np.inf
            for iteration in range(max_iter):
                if time_budget is not None and time_budget() < 15:
                    break
                
                # Update U
                H_U = np.einsum('br,cr->bcr', V, W).reshape(Mt * Pt, rank)
                target_flat = target_t.reshape(Nt, Mt * Pt)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                H_V = np.einsum('ar,cr->acr', U, W).reshape(Nt * Pt, rank)
                target_v = target_t.transpose(1, 0, 2).reshape(Mt, Nt * Pt)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                H_W = np.einsum('ar,br->abr', U, V).reshape(Nt * Mt, rank)
                target_w = target_t.transpose(2, 0, 1).reshape(Pt, Nt * Mt)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                T = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - target_t)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            # Early exit if we found exact solution
            if best_err < 1e-10:
                break
        
        if best_UVW is not None:
            return best_UVW, best_err
        return None, np.inf

    def build_subtensor_4x4():
        """Build the 4x4x4 sub-tensor for j=0,1,2,3."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for j in range(4):  # j = 0,1,2,3
                for k in range(6):
                    a = i * 5 + j
                    b = j * 6 + k
                    c = k * 4 + i
                    sub[a, b, c] = 1.0
        return sub
    
    def build_subtensor_1x6():
        """Build the 4x1x6 sub-tensor for j=4."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for k in range(6):
                a = i * 5 + 4  # j=4
                b = 4 * 6 + k  # j=4
                c = k * 4 + i
                sub[a, b, c] = 1.0
        return sub
    
    # Block partitioning strategy: split middle index j into 4+1
    # A is 4x5, B is 5x6, C is 4x6
    # Split j: j in 0..3 (4 values) and j=4 (1 value)
    # Part 1: 4x4x4 tensor (j=0,1,2,3) -> rank 7 via Strassen/ALS
    # Part 2: 4x1x6 tensor (j=4) -> rank 24 via schoolbook
    # Total: 7 + 24 = 31
    
    if time_remaining() > 50:
        sub1 = build_subtensor_4x4()
        # Use ALS to find rank-7 decomposition of sub1 with many restarts
        U1, V1, W1 = None, None, None
        best_err1 = np.inf
        
        for trial in range(15):
            if time_remaining() < 30:
                break
            result1 = als_decompose_subtensor(sub1, 7, max_iter=100, num_restarts=1, time_budget=time_remaining)
            if result1[0] is not None:
                U1, V1, W1 = result1[0]
                err1 = result1[1]
                if err1 < best_err1:
                    best_err1 = err1
                if err1 < 1e-10:
                    break
        
        if U1 is not None and best_err1 < 1e-8:
            # Decompose sub-tensor 2 (4x1x6 part) with rank 24 (schoolbook)
            U2 = np.zeros((N, 24))
            V2 = np.zeros((M, 24))
            W2 = np.zeros((P, 24))
            col = 0
            for i in range(4):
                for k in range(6):
                    a = i * 5 + 4
                    b = 4 * 6 + k
                    c = k * 4 + i
                    U2[a, col] = 1
                    V2[b, col] = 1
                    W2[c, col] = 1
                    col += 1
            
            # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2]
            U_combined = np.hstack([U1, U2])
            V_combined = np.hstack([V1, V2])
            W_combined = np.hstack([W1, W2])
            
            result_combined = {"U": U_combined, "V": V_combined, "W": W_combined}
            if verify(result_combined, atol=1e-8):
                rank_combined = U_combined.shape[1]
                if rank_combined < best_rank:
                    best = result_combined
                    best_rank = rank_combined
                    print(f"Block partitioning found rank {rank_combined}")

    # Final verification with tighter tolerance
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
