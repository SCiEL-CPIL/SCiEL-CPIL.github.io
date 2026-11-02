import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    # Block dimensions
    bn, bm, bp = 2, 3, 3
    num_blocks = p // bp # 3
    
    def solve_block(rank):
        """
        Attempts to find a rank decomposition for 2x3x3 multiplication.
        Returns U, V, W as lists or None if failed.
        """
        # Variables: U (6 x rank), V (9 x rank)
        # We optimize U and V. W is solved linearly.
        
        def loss(params):
            U = params[:6*rank].reshape(6, rank)
            V = params[6*rank:].reshape(9, rank)
            
            total_loss = 0.0
            # For each output (i, k), we have 3 equations (j=0,1,2)
            # and 'rank' unknowns w_r
            for i in range(bn):
                for k in range(bp):
                    # A is 3 x rank
                    A = np.zeros((3, rank))
                    for j in range(bm):
                        for r in range(rank):
                            A[j, r] = U[i*bm + j, r] * V[j*bp + k, r]
                    b = np.ones(3)
                    # Solve least squares A w = b
                    # w is rank x 1
                    # If A is 3xR, we want min ||Aw - b||^2
                    try:
                        w, residuals, rank_w, sv = np.linalg.lstsq(A, b, rcond=None)
                        total_loss += np.sum((A @ w - b)**2)
                    except:
                        return 1e10
            return total_loss

        # Initialize
        np.random.seed(42)
        # Start with random small values
        init_params = np.random.normal(0, 0.1, 6*rank + 9*rank)
        
        best_params = None
        best_loss = 1e10
        
        # Try a few restarts
        for restart in range(5):
            if restart > 0:
                init_params = np.random.normal(0, 0.5, 6*rank + 9*rank)
            
            # Use Nelder-Mead or L-BFGS-B
            # L-BFGS-B is better for smooth problems
            res = minimize(loss, init_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            if best_loss < 1e-6:
                break
                
        if best_loss > 1e-5:
            return None
            
        # Extract U, V
        U = best_params[:6*rank].reshape(6, rank)
        V = best_params[6*rank:].reshape(9, rank)
        
        # Compute W
        W = np.zeros((6, rank)) # 3*2 = 6
        for i in range(bn):
            for k in range(bp):
                A = np.zeros((3, rank))
                for j in range(bm):
                    for r in range(rank):
                        A[j, r] = U[i*bm + j, r] * V[j*bp + k, r]
                b = np.ones(3)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                # w is (rank, 1)
                for r in range(rank):
                    W[k*bn + i, r] = w[r]
                    
        # Verify
        # Check if U, V, W satisfy the identity
        # Sum_r U[i*3+j, r] V[j*3+k, r] W[k*2+i, r] == 1
        max_err = 0
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    val = 0
                    for r in range(rank):
                        val += U[i*bm+j, r] * V[j*bp+k, r] * W[k*bn+i, r]
                    if abs(val - 1) > max_err:
                        max_err = abs(val - 1)
                        
        if max_err > 1e-5:
            return None
            
        return U.tolist(), V.tolist(), W.tolist()

    # Search for minimal rank
    block_U = block_V = block_W = None
    found_rank = None
    
    for r in range(14, 19): # Try 14 to 18
        if time.time() > deadline - 20:
            break
        res = solve_block(r)
        if res is not None:
            block_U, block_V, block_W = res
            found_rank = r
            break
            
    if found_rank is None:
        # Fallback to schoolbook rank 18 for block
        block_U = [[0]*18 for _ in range(6)]
        block_V = [[0]*18 for _ in range(9)]
        block_W = [[0]*18 for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    block_U[i*3+j][col] = 1
                    block_V[j*3+k][col] = 1
                    block_W[k*2+i][col] = 1
                    col += 1
        found_rank = 18

    total_rank = found_rank * num_blocks
    
    # Construct full matrices
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for b in range(num_blocks):
        # Map block indices to full indices
        # U: i*3+j is same in block and full (since n,m same)
        # V: j*3+k in block -> j*9 + (b*3+k) in full
        # W: k*2+i in block -> (b*3+k)*2 + i in full
        
        for i in range(n):
            for j in range(m):
                for r in range(found_rank):
                    U_full[i*m+j][b*found_rank + r] = block_U[i*m+j][r]
                    
        for j in range(m):
            for k in range(bp):
                for r in range(found_rank):
                    V_full[j*p + (b*bp+k)][b*found_rank + r] = block_V[j*bp+k][r]
                    
        for k in range(bp):
            for i in range(n):
                for r in range(found_rank):
                    W_full[(b*bp+k)*n + i][b*found_rank + r] = block_W[k*n+i][r]

    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
