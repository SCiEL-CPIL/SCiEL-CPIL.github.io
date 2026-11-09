import numpy as np
import time
import sys
from fractions import Fraction
from itertools import combinations, product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nU = n * m  # 6
    nV = m * p  # 9
    nW = p * n  # 6
    
    # Build target tensor
    T = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook solution (rank 18)
    rank_sb = n * m * p
    U_sb = np.zeros((nU, rank_sb))
    V_sb = np.zeros((nV, rank_sb))
    W_sb = np.zeros((nW, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_and_round(U, V, W, T, atol=1e-6):
        """Check if decomposition is valid and round to half-integers."""
        for mat in [U, V, W]:
            rounded = np.round(mat * 2) / 2.0
            if np.max(np.abs(mat - rounded)) > 1e-3:
                return False
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=atol)
    
    def try_sparse_rank(R, max_patterns=500):
        """Try to find a valid decomposition of rank R by enumerating sparse support patterns."""
        if time.time() > deadline - 5:
            return False, None, None, None
        
        # For each rank R, we need to find support patterns for U, V, W
        # such that the bilinear system has a solution with half-integer entries.
        
        # Strategy: enumerate which (row, col) pairs are non-zero in each factor
        # For small R, this is feasible.
        
        # Generate candidate support patterns
        # For U: choose R columns from nU rows (each column has some non-zero entries)
        # For V: choose R columns from nV rows
        # For W: choose R columns from nW rows
        
        # Simpler approach: for each rank R, try random sparse patterns
        for attempt in range(max_patterns):
            if time.time() > deadline - 3:
                break
            
            # Random sparse pattern: each column has 1-2 non-zero entries
            U_support = []
            V_support = []
            W_support = []
            
            for r in range(R):
                # Each column has 1-2 non-zeros
                nU_nonzero = np.random.randint(1, 3)
                nV_nonzero = np.random.randint(1, 3)
                nW_nonzero = np.random.randint(1, 3)
                
                U_rows = np.random.choice(nU, size=nU_nonzero, replace=False)
                V_rows = np.random.choice(nV, size=nV_nonzero, replace=False)
                W_rows = np.random.choice(nW, size=nW_nonzero, replace=False)
                
                U_support.append(U_rows)
                V_support.append(V_rows)
                W_support.append(W_rows)
            
            # Build the linear system
            # Unknowns: U[i,r], V[j,r], W[k,r] for all (i,j,k) in support
            # Equations: for each (u_idx, v_idx, w_idx) with T[u_idx,v_idx,w_idx] = 1,
            #   sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r] = 1
            # This is nonlinear. Instead, fix some variables and solve linearly.
            
            # Alternative: use the fact that for fixed V and W, U is linear.
            # Or use alternating minimization with exact solves.
            
            # Initialize with random values on support
            U = np.zeros((nU, R))
            V = np.zeros((nV, R))
            W = np.zeros((nW, R))
            
            for r in range(R):
                for idx in U_support[r]:
                    U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in V_support[r]:
                    V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in W_support[r]:
                    W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Check if this pattern works
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-6):
                return True, U, V, W
            
            # Try alternating minimization: fix two factors, solve for the third
            for iteration in range(100):
                if time.time() > deadline - 2:
                    break
                
                # Solve for U given V, W
                # For each u_idx, we need: sum_r U[u_idx,r] * (sum over r where V,W are nonzero) = target
                # This is still nonlinear. Use gradient descent with exact rounding.
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.sum((T_recon - T)**2)
                
                if err < 1e-8:
                    break
                
                # Gradient step
                lr = 0.1
                dU = np.einsum('ijk,kr,jr->ir', T_recon - T, W, V)
                dV = np.einsum('ijk,ir,kr->jr', T_recon - T, U, W)
                dW = np.einsum('ijk,ir,jr->kr', T_recon - T, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Zero out non-support entries
                for r in range(R):
                    U[~np.isin(np.arange(nU), U_support[r]), r] = 0
                    V[~np.isin(np.arange(nV), V_support[r]), r] = 0
                    W[~np.isin(np.arange(nW), W_support[r]), r] = 0
                
                # Round to half-integers
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_recon, T, atol=1e-6):
                    return True, U, V, W
        
        return False, None, None, None
    
    def try_rank(R, max_restarts=30):
        """Try to find a valid decomposition of rank R with momentum-based optimization."""
        if time.time() > deadline - 5:
            return False, None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            if restart == 0:
                U = np.random.randn(nU, R) * 0.5
                V = np.random.randn(nV, R) * 0.5
                W = np.random.randn(nW, R) * 0.5
            elif restart == 1:
                U = np.random.choice([0, 1], size=(nU, R)) * 0.5
                V = np.random.choice([0, 1], size=(nV, R)) * 0.5
                W = np.random.choice([0, 1], size=(nW, R)) * 0.5
            elif restart == 2:
                U = np.random.randn(nU, R) * 0.1
                V = np.random.randn(nV, R) * 0.1
                W = np.random.randn(nW, R) * 0.1
            else:
                scale = 0.1 + 0.1 * restart / max_restarts
                U = np.random.randn(nU, R) * scale
                V = np.random.randn(nV, R) * scale
                W = np.random.randn(nW, R) * scale
            
            lr = 0.05
            momentum = 0.9
            best_err = np.inf
            stuck_count = 0
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(5000):
                if time.time() > deadline - 2:
                    break
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err_tensor = T_recon - T
                err = np.sum(err_tensor**2)
                
                if err < 1e-8:
                    break
                
                if err > best_err * 1.5:
                    lr *= 0.5
                elif err < best_err * 0.9:
                    lr = min(lr * 1.05, 0.08)
                    best_err = err
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if stuck_count > 500:
                    break
                
                dU = np.einsum('ijk,kr,jr->ir', err_tensor, W, V)
                dV = np.einsum('ijk,ir,kr->jr', err_tensor, U, W)
                dW = np.einsum('ijk,ir,jr->kr', err_tensor, U, V)
                
                vU = momentum * vU + (1 - momentum) * dU
                vV = momentum * vV + (1 - momentum) * dV
                vW = momentum * vW + (1 - momentum) * dW
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                if step % 500 == 0 and step > 0:
                    if check_and_round(U, V, W, T):
                        return True, U, V, W
            
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-5):
                if check_and_round(U, V, W, T):
                    return True, U, V, W
        
        return False, None, None, None
    
    # Try to find rank 17 with sparse pattern search first
    if time.time() < deadline - 20:
        success, U17, V17, W17 = try_sparse_rank(17, max_patterns=100)
        if not success:
            success, U17, V17, W17 = try_rank(17, max_restarts=40)
        if success:
            best_rank = 17
            best_U = U17
            best_V = V17
            best_W = W17
            print(f"Found rank 17 solution!", file=sys.stderr)
    
    # Try rank 16 if time permits
    if best_rank == 18 and time.time() < deadline - 15:
        success, U16, V16, W16 = try_sparse_rank(16, max_patterns=100)
        if not success:
            success, U16, V16, W16 = try_rank(16, max_restarts=30)
        if success:
            best_rank = 16
            best_U = U16
            best_V = V16
            best_W = W16
            print(f"Found rank 16 solution!", file=sys.stderr)
    
    # Convert to lists with half-integer rounding
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    R_final = len(U_list[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] for r in range(R_final))
                if abs(val - 1.0) > 1e-6:
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
