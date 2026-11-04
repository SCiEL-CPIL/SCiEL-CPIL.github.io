import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
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

    def get_strassen_4x4x4():
        """
        Return the Strassen decomposition for 4x4x4 matrix multiplication with rank 16.
        This uses the Winograd/Strassen algorithm structure.
        
        For 4x4 matrix multiplication, we can use a rank-16 decomposition.
        The idea is to split 4x4 into blocks and use Strassen-like formulas.
        
        Actually, for 4x4, we can use the fact that 4x4 = 2*(2x2) and apply
        Strassen's 7-multiplication algorithm twice, but that gives rank 49.
        
        A better approach: use the known rank-16 algorithm for 4x4.
        One such algorithm uses the following structure:
        
        For multiplying A (4x4) by B (4x4) to get C (4x4):
        We can use a decomposition based on the tensor rank of 4x4x4.
        
        The standard Strassen for 2x2 has rank 7.
        For 4x4, we can view it as 2x2 blocks of 2x2, giving rank 7^2 = 49.
        
        But there are better algorithms. Let me use a different approach:
        Use the fact that we only need to multiply 2x5 by 5x6, not full 4x4.
        
        Alternative: Use a structured ansatz based on the tensor structure.
        """
        pass

    def try_strassen_projection():
        """
        Try to use Strassen-like structure to find a lower-rank decomposition.
        
        Strategy: 
        1. Start with the 4x4x4 Strassen decomposition (rank 16 or similar).
        2. Project it down to the 2x5x6 sub-tensor.
        3. Check if the projection gives a valid decomposition.
        
        For 2x5x6, we can think of it as:
        - A is 2x5 (rows 0-1, cols 0-4)
        - B is 5x6 (rows 0-4, cols 0-5)
        - C is 2x6 (rows 0-1, cols 0-5)
        
        We can embed this in a 4x4x4 space by:
        - Padding A to 4x4: rows 0-1 used, rows 2-3 zero; cols 0-4 used, col 3 zero
        - Padding B to 4x4: rows 0-4 used... wait, 5 > 4.
        
        This doesn't work directly. Let me try a different approach.
        
        Alternative: Use a greedy/structured search based on the tensor structure.
        """
        
        # Let's try a different structural approach:
        # Use the fact that the tensor has a specific structure and try
        # to find a decomposition using a combination of:
        # 1. Sparse patterns
        # 2. Small integer values
        # 3. Exploiting the rectangular shape
        
        # For 2x5x6, the tensor has 60 non-zero entries.
        # We want to decompose it into rank-R simple tensors.
        
        # Let's try a structured approach:
        # Fix W to have a specific sparse pattern, then solve for U and V.
        
        # Pattern 1: W has 1s on the "diagonal" for each (k,i) pair
        # For each (k,i), set W[k*n+i, r] = 1 for some r, 0 otherwise.
        # Then U[i*m+j, r] * V[j*p+k, r] must sum to 1 for each (i,j,k).
        
        # This is equivalent to finding a matrix factorization.
        
        # Let's try: for each (k,i), assign a unique rank index r.
        # There are p*n = 12 such pairs, so rank >= 12.
        # But we need to satisfy all (i,j,k) constraints.
        
        # Actually, let's try a simpler approach: use the schoolbook but
        # try to merge some terms.
        
        return None

    def try_structured_search():
        """
        Try a structured search based on the tensor structure.
        
        Approach:
        1. For each (k,i) pair, we need sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        2. If we fix W[k*n+i, r] = 1 for one specific r per (k,i), and 0 otherwise,
           then we need U[i*m+j, r] * V[j*p+k, r] = 1 for all j.
        3. This means U[i*m+j, r] and V[j*p+k, r] must be inverses.
        4. For this to work for all j, we need U[i*m+j, r] = a_j and V[j*p+k, r] = 1/a_j.
        5. But a_j must be the same for all (i,k) pairs that share the same r.
        
        This is getting complex. Let's try a different approach.
        
        Use optimization with a better initialization based on the structure.
        """
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            R = len(params) // (nm + mp + pn)
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(pn, R)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err += (val - 1.0) ** 2
            return err

        # Try ranks from 30 to 50
        best_result = None
        best_rank = 60
        
        for R in range(30, 51):
            if time.time() > deadline - 30:
                break
            
            # Multiple random starts
            for start in range(3):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(start * 1000 + R)
                U_init = np.random.uniform(-1, 1, (nm, R))
                V_init = np.random.uniform(-1, 1, (mp, R))
                W_init = np.random.uniform(-1, 1, (pn, R))
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    
                    if res.fun < 1e-4:
                        U_opt = res.x[:nm*R].reshape(nm, R)
                        V_opt = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                        W_opt = res.x[nm*R+mp*R:].reshape(pn, R)
                        
                        # Try to round
                        for round_type in ['int', 'half']:
                            if round_type == 'int':
                                U_r = np.round(U_opt)
                                V_r = np.round(V_opt)
                                W_r = np.round(W_opt)
                            else:
                                U_r = np.round(U_opt * 2) / 2
                                V_r = np.round(V_opt * 2) / 2
                                W_r = np.round(W_opt * 2) / 2
                            
                            if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                                return U_r, V_r, W_r, R
                            
                            # Also check if the unrounded solution is close enough
                            if check_decomposition(U_opt, V_opt, W_opt, tol=1e-9):
                                # Keep as is if already exact
                                return U_opt, V_opt, W_opt, R
                except Exception:
                    continue
        
        return None

    def try_rank_optimize(R, num_starts=5):
        """
        Attempt to find a decomposition of rank R using optimization.
        """
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(pn, R)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err += (val - 1.0) ** 2
            return err

        for start in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(start)
            U_init = np.random.uniform(-1, 1, (nm, R))
            V_init = np.random.uniform(-1, 1, (mp, R))
            W_init = np.random.uniform(-1, 1, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:nm*R].reshape(nm, R)
                    V_opt = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    if check_decomposition(U_int, V_int, W_int, tol=1e-9):
                        return U_int, V_int, W_int
                        
                    U_half = np.round(U_opt * 2) / 2
                    V_half = np.round(V_opt * 2) / 2
                    W_half = np.round(W_opt * 2) / 2
                    
                    if check_decomposition(U_half, V_half, W_half, tol=1e-9):
                        return U_half, V_half, W_half
            except Exception:
                continue
                
        return None

    def try_rank_random(R, num_attempts=200):
        """Try random search for a valid decomposition of rank R."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            W = np.random.randint(-1, 2, size=(pn, R))
            
            if check_decomposition(U, V, W):
                return U, V, W
        return None

    # Strategy:
    # 1. Try structured search for ranks 30-50
    # 2. Try optimization for ranks 59 down to 50
    # 3. Try random search for ranks 59 down to 50
    # 4. Return schoolbook (rank 60)
    
    # First, try the structured approach
    result = try_structured_search()
    if result is not None:
        U, V, W, R = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Then try optimization
    for R in range(59, 49, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank_optimize(R, num_starts=2)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        if time.time() > deadline - 10:
            break
        
        result = try_rank_random(R, num_attempts=50)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
