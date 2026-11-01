"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients of tensor_error w.r.t. U, V, W efficiently."""
        r = U.shape[1]
        R = compute_R(U, V, W)
        E = R - target  # error tensor
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            M_U = np.einsum('abc,c->ab', E, W[:, c])
            grad_U[:, c] = 2.0 * (M_U @ V[:, c])
            
            N_V = np.einsum('abc,a->bc', E, U[:, c])
            grad_V[:, c] = 2.0 * (N_V @ W[:, c])
            
            UV_outer = np.outer(U[:, c], V[:, c])  # shape (nm, mp)
            grad_W[:, c] = 2.0 * np.einsum('abc,ab->c', E, UV_outer)
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, max_iters=500, lr_init=0.1):
        """Optimize U, V, W to minimize tensor error using gradient descent with momentum."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        # Momentum
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.9
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            # Update velocities
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            # Update
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                # Decay learning rate
                lr *= 0.995
            else:
                # Roll back and reduce learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        return U, V, W, prev_error
    
    def polish(U, V, W, max_iters=1000, lr_init=0.01):
        """High-precision polish pass with tighter tolerance and more iterations."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.95
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        return U, V, W, prev_error
    
    def strassen_like_block(rank, attempt=0):
        """
        Strassen-like recursive block decomposition strategy.
        Treat 2x3 as column of 2x1 blocks and 3x7 as row of 1x7 blocks.
        Use m=3 as small intermediate dimension.
        
        Key insight: Group operations to exploit structure and reduce rank.
        """
        # For 2x3 * 3x7, we can think of this as:
        # A is 2x3, B is 3x7
        # Split B into three 1x7 rows: B1, B2, B3
        # Result C = A1*B1 + A2*B2 + A3*B3 where Ai are rows of A
        
        # Strassen-like approach: use precomputed linear combinations
        # to reduce the number of scalar multiplications needed
        
        # For 2x3 * 3x7, the key is that m=3 is small
        # We can potentially use a decomposition that exploits this
        
        # Strategy: Try to find a decomposition where the intermediate
        # dimension m=3 allows for rank reduction
        
        # Let's try a structured approach:
        # Use the fact that we can precompute some linear combinations
        
        rng = np.random.RandomState(42 + attempt)
        
        # Initialize with structured pattern
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # For each rank component, try to capture a portion of the tensor
        # The target has 1s at positions (i*m+j, j*p+k, k*n+i)
        
        # Try different structured patterns
        if attempt == 0:
            # Diagonal dominant
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                if c < n*m*p:
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
        elif attempt == 1:
            # Strassen-inspired: use linear combinations
            # For the 3x3 intermediate, use Strassen-like formulas
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                if c < n*m*p:
                    # Use half-integer coefficients for Strassen-like reduction
                    U[i*m+j, c] = 1.0 if (i+j+k) % 2 == 0 else 0.5
                    V[j*p+k, c] = 1.0 if (i+j+k) % 2 == 0 else 0.5
                    W[k*n+i, c] = 1.0
        elif attempt == 2:
            # Block structure: group by k
            for c in range(rank):
                k = c % p
                i = (c // p) % n
                j = (c // (n*p)) % m
                if c < n*m*p:
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    if c >= n*m*p//2:
                        U[i*m+j, c] += 0.5
        else:
            # Randomized structured
            for c in range(rank):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                U[i*m+j, c] = rng.choice([0.5, 1.0])
                V[j*p+k, c] = rng.choice([0.5, 1.0])
                W[k*n+i, c] = rng.choice([0.5, 1.0])
        
        return U, V, W
    
    def algebraic_solve(rank, num_attempts=5):
        """
        Use algebraic structure: fix W to have simple structure, solve linear system for U, V.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline - 20:
                break
            
            # Create a structured W matrix
            W = np.zeros((pn, rank))
            
            if attempt == 0:
                for c in range(rank):
                    W[c % pn, c] = 1.0
            elif attempt == 1:
                for c in range(rank):
                    W[(c * 3) % pn, c] = 1.0
                    W[(c * 3 + 1) % pn, c] = 0.5
            elif attempt == 2:
                rng = np.random.RandomState(42 + attempt)
                for c in range(rank):
                    idx = rng.choice(pn, size=min(3, pn), replace=False)
                    for i in idx:
                        W[i, c] = rng.choice([0.5, 1.0, -0.5, -1.0])
            elif attempt == 3:
                for c in range(rank):
                    for i in range(3):
                        W[(c + i * (rank // 3)) % pn, c] = 1.0 if i == 0 else 0.5
            else:
                rng = np.random.RandomState(100 + attempt)
                for c in range(rank):
                    W[c % pn, c] = 1.0
                    W[(c + 1) % pn, c] = 0.5
            
            rng = np.random.RandomState(200 + attempt)
            V = rng.randn(mp, rank) * 0.5
            
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                M = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M[b * pn + c, :] = V[b, :] * W[c, :]
                
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    rhs = target[a, :, :].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        U_new[a, :] = np.zeros(rank)
                
                N_mat = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        N_mat[a * pn + c, :] = U_new[a, :] * W[c, :]
                
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    rhs = target[:, b, :].flatten()
                    try:
                        V_new[b, :] = np.linalg.lstsq(N_mat, rhs, rcond=None)[0]
                    except:
                        V_new[b, :] = np.zeros(rank)
                
                U = U_new
                V = V_new
                
                err = tensor_error(U, V, W)
                if err < 1e-10:
                    break
            
            U_rounded = round_to_half_integer(U)
            V_rounded = round_to_half_integer(V)
            W_rounded = round_to_half_integer(W)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            
            U_pol, V_pol, W_pol, err = polish(U_rounded, V_rounded, W_rounded, max_iters=200, lr_init=0.005)
            U_pol = round_to_half_integer(U_pol)
            V_pol = round_to_half_integer(V_pol)
            W_pol = round_to_half_integer(W_pol)
            
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol
        
        return None, None, None
    
    def try_rank(rank, num_attempts=5):
        """Try to find a valid decomposition for a given rank using multiple strategies."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Strategy 1: Strassen-like block decomposition
        for attempt in range(min(4, num_attempts)):
            if time.time() > deadline - 25:
                break
            
            U_init, V_init, W_init = strassen_like_block(rank, attempt)
            U_opt, V_opt, W_opt, final_error = optimize(U_init, V_init, W_init, max_iters=150, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # Strategy 2: Random init + gradient descent
        for attempt in range(min(2, num_attempts)):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(hash((rank, attempt, 1)) % 2**32)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=150, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # Strategy 3: Algebraic approach
        if time.time() < deadline - 25:
            U_alg, V_alg, W_alg = algebraic_solve(rank, num_attempts=3)
            if U_alg is not None:
                return U_alg, V_alg, W_alg
        
        # Strategy 4: Start from schoolbook and try to reduce rank
        if rank < n * m * p and time.time() < deadline - 25:
            U_sb, V_sb, W_sb = schoolbook()
            U_red = U_sb[:, :rank]
            V_red = V_sb[:, :rank]
            W_red = W_sb[:, :rank]
            
            U_opt, V_opt, W_opt, final_error = optimize(U_red, V_red, W_red, max_iters=150, lr_init=0.05)
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # If we have a best candidate, try to polish and round
        if best_U is not None and best_error < 1.0:
            U_pol, V_pol, W_pol, _ = polish(best_U, best_V, best_W, max_iters=300, lr_init=0.005)
            U_pol = round_to_half_integer(U_pol)
            V_pol = round_to_half_integer(V_pol)
            W_pol = round_to_half_integer(W_pol)
            
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol
        
        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high
    for rank in range(25, 42):
        if time.time() > deadline - 25:
            break
        
        U, V, W = try_rank(rank, num_attempts=5)
        
        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
