"""Matrix multiplication 2x4 x 4x6 rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The tensor T[i,j,k] = delta for matrix mult: T[i,j,k] = 1 if i,j,k valid
    # Actually the tensor is T[i,j,k] where the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 for all valid (i,j,k)
    # But the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read: W[k*n+i, r] - so W is indexed by (k,i)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a 3-tensor with entries all 1 (size n x m x p = 2 x 4 x 6 = 48 entries)
    
    # Known lower bounds: For 2x4x6, the minimum rank is at least max(n*m, m*p, p*n) = max(8,24,12) = 24
    # But that's a very weak bound. Better bounds exist.
    # Actually, for matrix multiplication M_{n,m,p}, lower bound is max(nm, mp, pn) is not right.
    # The trivial lower bound is max(nm, mp, pn) / something... 
    # Actually rank >= max(n*m, m*p, p*n) is NOT correct.
    # rank >= n*m, rank >= m*p, rank >= p*n are all valid since projecting gives those.
    # So rank >= max(8, 24, 12) = 24.
    
    # Let's try to find a decomposition with rank < 48.
    # Strategy: Use random initialization + optimization to minimize ||T_decomposed - T||_F^2
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, _ = schoolbook()
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    # Try various ranks from low to high
    # Lower bound is 24, so try 24, 25, 26, ... up to maybe 40
    # For each rank, do random restarts with optimization
    
    def decompose_residual(U, V, W, rank):
        """Compute the decomposed tensor and return residual norm."""
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape to 3D: U3[i,j,c], V3[j,k,c], W3[k,i,c]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # = einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        T_true = np.ones((n, m, p))
        return np.linalg.norm(T_dec - T_true)
    
    def try_rank(rank, max_iters=2000, n_restarts=5, lr=0.01):
        """Try to find a valid decomposition of given rank."""
        nonlocal best_result, best_rank
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                return
            
            # Random initialization
            rng = np.random.RandomState(restart * 1000 + rank)
            scale = 1.0 / (rank ** 0.5)
            U = rng.randn(nm, rank) * scale
            V = rng.randn(mp, rank) * scale
            W = rng.reshape(pn, rank) * scale
            
            # Coordinate descent / gradient-based optimization
            # Objective: f = ||einsum(U3,V3,W3) - 1||_F^2
            # Use simple gradient descent with adaptive step
            
            best_local = decompose_residual(U, V, W, rank)
            
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    break
                
                # Compute gradients
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                
                T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                residual = T_dec - 1.0
                
                # Gradient of f = sum_{ijk} residual[i,j,k]^2
                # df/dU[i,j,c] = 2 * sum_{k} residual[i,j,k] * V3[j,k,c] * W3[k,i,c]
                # = 2 * einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
                dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm < 1e-12:
                    break
                
                # Simple: use fixed small step
                step = lr / max(1.0, grad_norm * 0.1)
                
                U -= step * dU
                V -= step * dV
                W -= step * dW
                
                new_res = decompose_residual(U, V, W, rank)
                if new_res < best_local:
                    best_local = new_res
                    if best_local < 1e-6:
                        # Found a good solution, try to rationalize
                        return _try_rationalize(U, V, W, rank)
                else:
                    # Revert and reduce step
                    U += step * dU
                    V += step * dV
                    W += step * dW
                    lr *= 0.5
                    if lr < 1e-8:
                        break
            
            if best_local < 1e-6:
                return _try_rationalize(U, V, W, rank)
        
        return None
    
    def _try_rationalize(U, V, W, rank):
        """Try to snap to rational values and verify exactly."""
        nonlocal best_result, best_rank
        
        # Scale to make entries O(1)
        # Try to find rational approximations
        # Use continued fractions or just round to nearest 1/2, 1/3, 1/4...
        
        # First, check if current solution is already close to rational
        # Try denominators 1, 2, 3, 4, 6
        for denom in [1, 2, 3, 4, 6]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            U3 = U_r.reshape(n, m, rank)
            V3 = V_r.reshape(m, p, rank)
            W3 = W_r.reshape(p, n, rank)
            T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            if np.allclose(T_dec, np.ones((n, m, p)), atol=1e-10):
                # Verify exactly with fractions
                if _verify_exact(U_r, V_r, W_r, rank, denom):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r, rank)
                    return True
        
        # Try with larger denominators
        for denom in [12, 24]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            U3 = U_r.reshape(n, m, rank)
            V3 = V_r.reshape(m, p, rank)
            W3 = W_r.reshape(p, n, rank)
            T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            if np.allclose(T_dec, np.ones((n, m, p)), atol=1e-8):
                if _verify_exact(U_r, V_r, W_r, rank, denom):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r, rank)
                    return True
        
        return False
    
    def _verify_exact(U, V, W, rank, denom):
        """Verify exact decomposition using fractions."""
        # Convert to fractions
        Uf = [[Fraction(int(round(U[i,c] * denom)), denom) for c in range(rank)] for i in range(nm)]
        Vf = [[Fraction(int(round(V[i,c] * denom)), denom) for c in range(rank)] for i in range(mp)]
        Wf = [[Fraction(int(round(W[i,c] * denom)), denom) for c in range(rank)] for i in range(pn)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try ranks from lower bound upward
    # For 2x4x6, known results suggest rank might be around 28-36
    # Let's try a range
    for rank in range(24, 42):
        if time.time() > DEADLINE:
            break
        
        result = try_rank(rank, max_iters=1500, n_restarts=3, lr=0.05)
        if result:
            # Found a valid decomposition
            break
    
    # Also try a different approach: structured decomposition
    # Exploit the fact that matrix mult tensor has special structure
    # Try "split" decompositions
    
    if time.time() < DEADLINE - 5:
        _try_structured()
    
    # Return best result
    U_best, V_best, W_best, r_best = best_result
    
    # Convert to lists
    U_list = [[U_best[i, c] for c in range(r_best)] for i in range(nm)]
    V_list = [[V_best[i, c] for c in range(r_best)] for i in range(mp)]
    W_list = [[W_best[i, c] for c in range(r_best)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_structured():
    """Try structured decompositions exploiting matrix mult structure."""
    nonlocal best_result, best_rank, DEADLINE, n, m, p, nm, mp, pn
    
    # One approach: use the fact that we can "share" computations
    # For example, if we fix certain patterns in U, V, W
    
    # Try a decomposition where we use "block" structure
    # A[i,j] * B[j,k]: we can think of this as (n*m) x (m*p) bilinear map to (p*n)
    
    # Try random structured initialization
    for trial in range(5):
        if time.time() > DEADLINE:
            return
        
        rng = np.random.RandomState(42 + trial)
        
        # Try rank = 32 (middle ground)
        rank = 30 + trial
        
        # Initialize with some structure: make U, V, W have rows that are
        # random but with some zeros to exploit sparsity
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5
        
        # Add a "bias" structure: some columns are identity-like
        for c in range(min(rank, nm)):
            U[c, c] += 1.0
        
        # Optimize
        lr = 0.02
        for it in range(3000):
            if time.time() > DEADLINE:
                break
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            residual = T_dec - 1.0
            
            dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
            dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
            dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
            
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-10:
                break
            
            step = lr / max(1.0, grad_norm * 0.05)
            
            U_new = U - step * dU
            V_new = V - step * dV
            W_new = W - step * dW
            
            U3n = U_new.reshape(n, m, rank)
            V3n = V_new.reshape(m, p, rank)
            W3n = W_new.reshape(p, n, rank)
            T_dec_n = np.einsum('ijc,jkc,kic->ijk', U3n, V3n, W3n)
            new_res = np.linalg.norm(T_dec_n - 1.0)
            old_res = np.linalg.norm(residual)
            
            if new_res < old_res:
                U, V, W = U_new, V_new, W_new
            else:
                lr *= 0.7
                if lr < 1e-10:
                    break
        
        # Check and try to rationalize
        res = np.linalg.norm(np.einsum('ijc,jkc,kic->ijk', 
            U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank)) - 1.0)
        
        if res < 0.1:
            # Try rationalization
            for denom in [1, 2, 3, 4, 6, 12]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                T_dec = np.einsum('ijc,jkc,kic->ijk', 
                    U_r.reshape(n,m,rank), V_r.reshape(m,p,rank), W_r.reshape(p,n,rank))
                if np.allclose(T_dec, 1.0, atol=1e-10):
                    # Verify exactly
                    if _verify_exact(U_r, V_r, W_r, rank, denom):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = (U_r, V_r, W_r, rank)
                        return

# Entry point
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def decompose_residual(U, V, W, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.linalg.norm(T_dec - np.ones((n, m, p)))
    
    def verify_exact(U, V, W, rank, denom):
        Uf = [[Fraction(int(round(U[i,c]*denom)), denom) for c in range(rank)] for i in range(nm)]
        Vf = [[Fraction(int(round(V[i,c]*denom)), denom) for c in range(rank)] for i in range(mp)]
        Wf = [[Fraction(int(round(W[i,c]*denom)), denom) for c in range(rank)] for i in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def try_rank_opt(rank, n_restarts=3, max_iters=2000):
        nonlocal best_U, best_V, best_W, best_rank
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                return False
            
            rng = np.random.RandomState(restart * 7919 + rank * 104729)
            scale = 1.0 / np.sqrt(rank)
            U = rng.randn(nm, rank) * scale
            V = rng.randn(mp, rank) * scale
            W = rng.randn(pn, rank) * scale
            
            lr = 0.05
            best_res = decompose_residual(U, V, W, rank)
            
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    return False
                
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                residual = T_dec - 1.0
                cur_res = np.linalg.norm(residual)
                
                if cur_res < 1e-7:
                    break
                
                dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
                dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
                
                gn = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gn < 1e-12:
                    break
                
                step = lr / max(1.0, gn * 0.1)
                U_new = U - step * dU
                V_new = V - step * dV
                W_new = W - step * dW
                
                new_res = decompose_residual(U_new, V_new, W_new, rank)
                if new_res < cur_res:
                    U, V, W = U_new, V_new, W_new
                    best_res = new_res
                else:
                    lr *= 0.5
                    if lr < 1e-10:
                        break
            
            if best_res < 1e-6:
                # Try to rationalize
                for denom in [1, 2, 3, 4, 6, 12, 24]:
                    U_r = np.round(U * denom) / denom
                    V_r = np.round(V * denom) / denom
                    W_r = np.round(W * denom) / denom
                    T_dec = np.einsum('ijc,jkc,kic->ijk', 
                        U_r.reshape(n,m,rank), V_r.reshape(m,p,rank), W_r.reshape(p,n,rank))
                    if np.allclose(T_dec, 1.0, atol=1e-9):
                        if verify_exact(U_r, V_r, W_r, rank, denom):
                            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank
                            return True
        
        return False
    
    # Try ranks
    for rank in range(24, 40):
        if time.time() > DEADLINE - 10:
            break
        if try_rank_opt(rank):
            break
    
    # Convert to list format
    r = best_rank
    U_list = [[float(best_U[i, c]) for c in range(r)] for i in range(nm)]
    V_list = [[float(best_V[i, c]) for c in range(r)] for i in range(mp)]
    W_list = [[float(best_W[i, c]) for c in range(r)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
