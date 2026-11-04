"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=8):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    # Try denominators 1, 2, 4, 8
    for denom in [1, 2, 4, 8]:
        step = 1.0 / denom
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _als_polish(U, V, W, T, n, m, p, iterations=20):
    """
    Polish the solution using Alternating Least Squares.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for _ in range(iterations):
        # Solve for U
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed V, W:
        # U = T_mat @ V_mat @ (V_mat^T V_mat)^-1
        T_mat = T.reshape(nm, mp*pn)
        
        V_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                V_mat[b*pn+c, :] = V[b, :] * W[c, :]
                
        try:
            VtV = V_mat.T @ V_mat
            VtV += 1e-6 * np.eye(r)
            U = T_mat @ V_mat @ np.linalg.inv(VtV)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for V
        T_mat_v = T.reshape(mp, nm*pn)
        U_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                U_mat[a*pn+c, :] = U[a, :] * W[c, :]
        try:
            UtU = U_mat.T @ U_mat
            UtU += 1e-6 * np.eye(r)
            V = T_mat_v @ U_mat @ np.linalg.inv(UtU)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for W
        T_mat_w = T.reshape(pn, nm*mp)
        UW_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UW_mat[a*mp+b, :] = U[a, :] * V[b, :]
        try:
            UWtUW = UW_mat.T @ UW_mat
            UWtUW += 1e-6 * np.eye(r)
            W = T_mat_w @ UW_mat @ np.linalg.inv(UWtUW)
        except np.linalg.LinAlgError:
            pass

    return U, V, W

def _laderman_block_decomp(n, m, p):
    """
    Construct a candidate decomposition using block structure.
    Partition A (n x m) into n blocks of (1 x m) and B (m x p) into m blocks of (1 x p) ? No.
    Standard block mult:
    A is n x m. B is m x p.
    If we split A into rows and B into cols?
    Actually, let's look at the structure of the tensor.
    T[i,j,k] = 1 if indices match.
    
    The strategy suggests:
    Partition 2x5 into 1x2 blocks? 5 is not divisible by 2.
    Partition 2x5 into 1x1, 1x1, 1x1, 1x1, 1x1? (5 blocks of 1x1)
    Partition 5x6 into 1x1...1x1? (5 blocks)
    This is just schoolbook.
    
    Let's try a different block partition.
    A: 2x5. Split into 2 blocks of (1 x 2.5)? No.
    Split A into 1x5 and 1x5.
    Split B into 5x3 and 5x3.
    Then C = A1 * B1 + A2 * B2.
    A1 is 1x5, B1 is 5x3 -> C1 is 1x3.
    A2 is 1x5, B2 is 5x3 -> C2 is 1x3.
    C = [C1; C2] is 2x3.
    But we need 2x6. So B is 5x6. Split B into 5x3 and 5x3.
    A is 2x5. It multiplies both.
    So C = A * B_left + A * B_right.
    This is just two multiplications of 2x5 by 5x3.
    Rank of 2x5x3?
    Schoolbook: 2*5*3 = 30.
    Two of them: 60.
    
    Is there a better way?
    Maybe use the fact that n=2 is small.
    C[i,k] = sum_j A[i,j] B[j,k].
    
    Let's stick to the general search but initialize with structured guesses if possible.
    For now, just return None to indicate no special block found, or use the schoolbook as the "block" result.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
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
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 30 to 59
    for rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
            
        found = False
        max_restarts = 20  # Increase restarts slightly
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            # Use a mix of random and structured initialization
            if restart == 0:
                # Sparse random init
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
            else:
                # Standard random
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
            
            # Adam Optimizer
            v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
            beta1, beta2 = 0.9, 0.999
            lr = 0.01
            t = 0
            best_loss = np.inf
            
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    if loss < 1e-10:
                        break
                        
                # Gradients
                gU = -np.einsum('abc,br,cr->ar', R, V, W)
                gV = -np.einsum('abc,ar,cr->br', R, U, W)
                gW = -np.einsum('abc,ar,br->cr', R, U, V)
                
                grads = [gU, gV, gW]
                params = [U, V, W]
                
                t += 1
                for i in range(3):
                    m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
                    m_hat = m_[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    
                U = params[0]
                V = params[1]
                W = params[2]
                
                if step % 50 == 0 and step > 0:
                    lr *= 0.5
                
                if step % 50 == 0 and loss < 1e-6:
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        found = True
                        break
                        
            if found:
                # Polish with ALS
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=10)
                if _verify_fast(U, V, W, T, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    _try_rational_snapping(U, V, W, T)
                    break

        if found:
            break

    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
