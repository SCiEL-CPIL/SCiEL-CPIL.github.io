import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Step 1: Build guaranteed rank-11 solution (Strassen 2x2x2 + schoolbook 2x1x2)
    # Strassen for C = A * B where A is 2x2, B is 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A11 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 2x2 * 2x3:
    # A is 2x2, B is 2x3 = [B_left(2x2) | B_right(2x1)]
    # C is 2x3 = [C_left(2x2) | C_right(2x1)]
    # C_left = A * B_left (use Strassen, rank 7)
    # C_right = A * B_right (schoolbook, rank 4)
    
    # Strassen factors for C_left:
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices: B00=0, B01=1, B10=2, B11=3
    # W indices: C00=0, C01=1, C10=2, C11=3
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, 1, 0],   # A00
        [0, 1, 0, 0, 0, 0, 1],   # A01
        [0, 0, 1, 1, 1, 0, 0],   # A10
        [1, 0, 0, 1, 0, -1, -1]  # A11
    ])
    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, -1, 1, 0, 1, 0],  # B01
        [1, 0, 1, 0, 1, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ])
    strassen_W = np.array([
        [1, 0, 0, 1],   # C00
        [0, 1, 0, 0],   # C01
        [0, 0, 1, 0],   # C10
        [1, 0, 0, 1]    # C11
    ])
    
    # Schoolbook for C_right = A * B_right where B_right is 2x1
    # B_right indices: Br0=4, Br1=5
    # C_right indices: Cr0=4, Cr1=5
    # 4 products: A00*Br0, A01*Br0, A10*Br0, A11*Br0 for Cr0
    #               A00*Br1, A01*Br1, A10*Br1, A11*Br1 for Cr1
    
    school_U = np.array([
        [1, 0, 0, 0],   # A00
        [0, 1, 0, 0],   # A01
        [0, 0, 1, 0],   # A10
        [0, 0, 0, 1]    # A11
    ])
    school_V = np.array([
        [1, 0, 1, 0],   # Br0 (for Cr0)
        [0, 1, 0, 1],   # Br1 (for Cr1)
        [0, 0, 1, 0],   # Br0 (for Cr0) - wait, this is wrong
        [0, 0, 0, 1]    # Br1 (for Cr1)
    ])
    # Actually for 2x2 * 2x1:
    # Cr0 = A00*Br0 + A01*Br0  (no, Cr0 = A00*Br0 + A01*Br0? No)
    # C is 2x1, B_right is 2x1
    # Cr0 = A00*Br0 + A01*Br0? No.
    # C[i] = sum_j A[i,j] * B_right[j]
    # Cr0 = A00*Br0 + A01*Br0? No, B_right is column vector [Br0, Br1]
    # Cr0 = A00*Br0 + A01*Br0? No.
    # Cr0 = A[0,0]*Br[0] + A[0,1]*Br[1]
    # Cr1 = A[1,0]*Br[0] + A[1,1]*Br[1]
    
    school_V = np.array([
        [1, 0],   # Br0 contributes to Cr0 and Cr1
        [0, 1],   # Br1 contributes to Cr0 and Cr1
        [1, 0],   # Br0 contributes to Cr0 and Cr1
        [0, 1]    # Br1 contributes to Cr0 and Cr1
    ])
    # Wait, let me redo this properly.
    # For C_right (2x1) = A (2x2) * B_right (2x1):
    # Cr[0] = A[0,0]*Br[0] + A[0,1]*Br[1]
    # Cr[1] = A[1,0]*Br[0] + A[1,1]*Br[1]
    # 4 rank-1 terms:
    # Term 1: A[0,0] * Br[0] -> Cr[0]
    # Term 2: A[0,1] * Br[1] -> Cr[0]
    # Term 3: A[1,0] * Br[0] -> Cr[1]
    # Term 4: A[1,1] * Br[1] -> Cr[1]
    
    school_U = np.array([
        [1, 0, 0, 0],   # A00
        [0, 1, 0, 0],   # A01
        [0, 0, 1, 0],   # A10
        [0, 0, 0, 1]    # A11
    ])
    school_V = np.array([
        [1, 0, 1, 0],   # Br0: appears in term1 (Cr0) and term3 (Cr1)
        [0, 1, 0, 1]    # Br1: appears in term2 (Cr0) and term4 (Cr1)
    ])
    school_W = np.array([
        [1, 1, 0, 0],   # Cr0: from term1 and term2
        [0, 0, 1, 1]    # Cr1: from term3 and term4
    ])
    
    # Combine into rank-11
    U11 = np.hstack([strassen_U, school_U])  # 4x11
    V11 = np.vstack([np.hstack([strassen_V, np.zeros((2, 4))]),
                      np.hstack([np.zeros((2, 7)), school_V])])  # 6x11
    W11 = np.vstack([strassen_W, school_W])  # 6x11
    
    # Verify rank-11 solution
    T_target = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T_target[i*2+j, j*3+k, k*2+i] = 1.0
    
    T11 = np.einsum('ir,jr,kr->ijk', U11, V11, W11)
    assert np.allclose(T11, T_target, atol=1e-10), "Rank-11 construction failed"
    
    best_U, best_V, best_W = U11.tolist(), V11.tolist(), W11.tolist()
    best_rank = 11
    
    # Step 2: Try to find rank-10 by allowing linear dependencies
    # Strategy: parameterize U, V, W with 10 columns and optimize
    # Use the rank-11 solution as initial guess by projecting
    
    np.random.seed(42)
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W, denom=2):
        U_rnd = np.round(U * denom) / denom
        V_rnd = np.round(V * denom) / denom
        W_rnd = np.round(W * denom) / denom
        T = compute_tensor(U_rnd, V_rnd, W_rnd)
        return np.allclose(T, T_target, atol=1e-8), U_rnd, V_rnd, W_rnd
    
    # Try rank 10 with various initializations
    for target_rank in [10]:
        if time.time() > deadline - 15.0:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                break
            
            # Different initialization strategies
            if attempt < 10:
                # Start from rank-11 and zero out one column
                U_init = U11[:, :target_rank].copy()
                V_init = V11[:, :target_rank].copy()
                W_init = W11[:, :target_rank].copy()
            elif attempt < 20:
                # Random with small perturbation around rank-11 structure
                U_init = U11[:, :target_rank].copy() + np.random.randn(4, target_rank) * 0.1
                V_init = V11[:, :target_rank].copy() + np.random.randn(6, target_rank) * 0.1
                W_init = W11[:, :target_rank].copy() + np.random.randn(6, target_rank) * 0.1
            elif attempt < 30:
                # Pure random
                scale = 0.5
                U_init = np.random.randn(4, target_rank) * scale
                V_init = np.random.randn(6, target_rank) * scale
                W_init = np.random.randn(6, target_rank) * scale
            else:
                # Strassen-like structure with perturbation
                U_init = strassen_U.copy() + np.random.randn(4, 7) * 0.05
                # Pad to 10 columns
                U_init = np.hstack([U_init, np.random.randn(4, 3) * 0.3])
                V_init = strassen_V.copy() + np.random.randn(4, 7) * 0.05
                V_init = np.vstack([V_init, np.random.randn(2, 7) * 0.3])
                V_init = np.hstack([V_init, np.random.randn(6, 3) * 0.3])
                W_init = strassen_W.copy() + np.random.randn(4, 7) * 0.05
                W_init = np.vstack([W_init, np.random.randn(2, 7) * 0.3])
                W_init = np.hstack([W_init, np.random.randn(6, 3) * 0.3])
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                if time.time() > deadline:
                    return 1e9
                U_flat = params[:4*target_rank]
                V_flat = params[4*target_rank:4*target_rank + 6*target_rank]
                W_flat = params[4*target_rank + 6*target_rank:]
                
                U = U_flat.reshape(4, target_rank)
                V = V_flat.reshape(6, target_rank)
                W = W_flat.reshape(6, target_rank)
                
                T_prime = compute_tensor(U, V, W)
                diff = T_prime - T_target
                return np.sum(diff**2)
            
            try:
                res = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < 1e-8:
                    U_flat = res.x[:4*target_rank]
                    V_flat = res.x[4*target_rank:4*target_rank + 6*target_rank]
                    W_flat = res.x[4*target_rank + 6*target_rank:]
                    U_num = U_flat.reshape(4, target_rank)
                    V_num = V_flat.reshape(6, target_rank)
                    W_num = W_flat.reshape(6, target_rank)
                    
                    # Try rounding to various denominators
                    for denom in [1, 2, 4, 8, 16]:
                        ok, U_rnd, V_rnd, W_rnd = verify(U_num, V_num, W_num, denom)
                        if ok:
                            best_U, best_V, best_W = U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist()
                            best_rank = target_rank
                            break
                        if time.time() > deadline - 5.0:
                            break
                    if best_rank == target_rank:
                        break
            except Exception:
                continue
    
    # Final verification
    U_final = np.array(best_U)
    V_final = np.array(best_V)
    W_final = np.array(best_W)
    T_final = compute_tensor(U_final, V_final, W_final)
    if not np.allclose(T_final, T_target, atol=1e-8):
        # Fallback to schoolbook rank 12
        U_sb = np.zeros((4, 12))
        V_sb = np.zeros((6, 12))
        W_sb = np.zeros((6, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_sb[i*2+j, col] = 1
                    V_sb[j*3+k, col] = 1
                    W_sb[k*2+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
