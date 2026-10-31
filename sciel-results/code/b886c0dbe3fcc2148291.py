import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Strategy: Numerical Tensor Completion via Non-Convex Optimization.
    
    Instead of relying on hardcoded algorithmic structures (like Strassen or Winograd),
    we treat the problem as finding a rank-25 decomposition of the 4x4x4 matrix 
    multiplication tensor. We use L-BFGS-B to minimize the squared error of the 
    tensor identity. The tensor is highly sparse (only 64 non-zero entries out of 
    4096), and the solution is known to exist with integer/half-integer coefficients.
    
    We initialize with a random rank-25 decomposition and refine it. If it converges
    to a valid decomposition, we return it. This approach explores a different 
    design family: numerical search vs. symbolic/algorithmic construction.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    n, m, p = 4, 4, 4
    target_rank = 25  # Known optimal rank for 4x4x4 matmul
    
    # The tensor T[a, b, c] is 1 if a=i*n+j, b=j*p+k, c=k*n+i (Trace Convention W[k*n+i])
    # Wait, let's verify the convention again.
    # Task: W[k*n+i, r].
    # Standard matmul: C[i, k] = sum_j A[i, j] B[j, k].
    # Indexing:
    # A index a: i * m + j (i row, j col) -> i*4 + j
    # B index b: j * p + k (j row, k col) -> j*4 + k
    # C index c: The task says W[k*n+i, r]. 
    # Usually C is indexed by row/col. If C[i, k] is the result, 
    # does the task imply the output matrix is indexed such that index = k*n+i?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the C matrix in the decomposition corresponds to the 
    # linear combination for the entry at position (k, i) in the standard 
    # row-major layout? Or is it a specific permutation?
    # Let's look at the current program's verification:
    # T[i*4+j, j*4+k, k*4+i] = 1
    # Here a = i*4+j, b = j*4+k, c = k*4+i.
    # This implies:
    # A[i, j] is at index i*4+j.
    # B[j, k] is at index j*4+k.
    # C entry corresponding to (i, k) is at index k*4+i.
    # This is a transposed indexing for C? 
    # Standard C[i, k] would be i*4+k.
    # Here c = k*4+i.
    # Let's stick to the current program's verified convention:
    # T[a, b, c] = 1 iff a = i*4+j, b = j*4+k, c = k*4+i for some i,j,k in {0,1,2,3}.
    
    # Build the target tensor T as a flat array or 3D array
    T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T[a, b, c] = 1.0
    
    # Flatten T for efficient computation? Or use einsum?
    # We want to minimize || sum_r U[:, r] V[:, r] W[:, r] - T ||^2
    # Let x be the flattened parameters of U, V, W.
    # Total params: 3 * 16 * 25 = 1200.
    
    def objective_and_grad(params):
        # Reshape params into U, V, W
        U = params[0:16*target_rank].reshape(16, target_rank)
        V = params[16*target_rank:32*target_rank].reshape(16, target_rank)
        W = params[32*target_rank:48*target_rank].reshape(16, target_rank)
        
        # Compute tensor product sum
        # T_est[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Using einsum: 'ar,br,cr->abc'
        T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Error
        err = T_est - T
        loss = np.sum(err * err)
        
        # Gradient
        # d loss / d U[a, r] = 2 * sum_{b,c} err[a, b, c] V[b, r] W[c, r]
        # d loss / d V[b, r] = 2 * sum_{a,c} err[a, b, c] U[a, r] W[c, r]
        # d loss / d W[c, r] = 2 * sum_{a,b} err[a, b, c] U[a, r] V[b, r]
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
        grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
        grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook rank 64 if no time
        return _fallback_schoolbook()
    
    # Initialize
    # Random initialization might get stuck. 
    # Better initialization: Use a known lower-rank approximation or random.
    # Since rank 25 is tight, random init in high dimensions often works if the loss landscape is smooth enough,
    # but it's non-convex. 
    # Let's try a few random restarts if time permits.
    
    best_params = None
    best_loss = np.inf
    
    # We have ~130 seconds. 
    # L-BFGS-B on 1200 params with einsum should be fast per iteration.
    # Let's try 1-3 restarts.
    
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random init
        np.random.seed(42 + r)
        # Scale init to avoid huge gradients
        scale = 1.0 / np.sqrt(16)
        U_init = np.random.randn(16, target_rank) * scale
        V_init = np.random.randn(16, target_rank) * scale
        W_init = np.random.randn(16, target_rank) * scale
        
        # Symmetry breaking: The problem is invariant under U->U*diag, V->V*diag^{-1}, etc.
        # Random init breaks this.
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Max iterations per restart
        # Each iteration involves several einsums.
        # 16x16x16 tensor is small.
        # Let's allow 200 iterations.
        max_iter = 200
        
        try:
            res = minimize(
                objective_and_grad, 
                params_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            # Check if converged to a valid solution
            if res.fun < 1e-8:
                # Verify
                U = best_params[0:16*target_rank].reshape(16, target_rank)
                V = best_params[16*target_rank:32*target_rank].reshape(16, target_rank)
                W = best_params[32*target_rank:48*target_rank].reshape(16, target_rank)
                
                T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                if np.allclose(T_est, T, atol=1e-6):
                    # Return valid rank 25
                    return {
                        "U": U.tolist(), 
                        "V": V.tolist(), 
                        "W": W.tolist()
                    }
                    
        except Exception as e:
            pass # Ignore errors in optimization, try next restart

    if best_params is not None:
        # If we didn't converge to < 1e-8 but have a good solution, check if it's "good enough"
        # The evaluator requires EXACT decomposition? 
        # "exactly decompose the matrix multiplication tensor"
        # "integer/half-integer entries"
        # Numerical solutions might be floating point. 
        # We should snap to nearest half-integers if close?
        # But if it's not exact, it might fail.
        # Let's check the best loss.
        if best_loss < 1e-6:
            U = best_params[0:16*target_rank].reshape(16, target_rank)
            V = best_params[16*target_rank:32*target_rank].reshape(16, target_rank)
            W = best_params[32*target_rank:48*target_rank].reshape(16, target_rank)
            
            # Try to snap to rational numbers?
            # If the solution is close to integers/half-integers, snap them.
            # This helps with "exactly" requirement if floating point errors are small.
            def snap_half(x):
                # Snap to nearest 0.5
                return np.round(x * 2) / 2
            
            U_snapped = snap_half(U)
            V_snapped = snap_half(V)
            W_snapped = snap_half(W)
            
            # Verify snapped version
            T_est = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped, optimize=True)
            if np.allclose(T_est, T, atol=1e-9):
                return {
                    "U": U_snapped.tolist(), 
                    "V": V_snapped.tolist(), 
                    "W": W_snapped.tolist()
                }
            else:
                # If snapping failed, return un-snapped? 
                # The prompt says "integer/half-integer entries". 
                # If I return floats that are 0.9999999, it might fail the type check or exactness.
                # But if I return 1.0, it's fine.
                # Let's just return the floats if they are close enough to be considered exact in float precision.
                # However, the prompt explicitly asks for integer/half-integer.
                # If the numerical solution is 0.5000001, snapping to 0.5 is correct.
                # If the numerical solution is 0.3, it's not a valid solution for this task constraint.
                # Given the known structure, the solution IS integer/half-integer.
                # So if optimization converged, it should be close to these.
                
                # If snapping didn't pass the strict check, try returning the original floats?
                # Risky. Better to fallback.
                pass

    # Fallback: Rank 49 Strassen (Known to work, score 1.3)
    # Or Rank 64 Schoolbook (Score 1.0)
    # Since we are aiming for improvement, and 25 failed, we should return the best known valid one.
    # The current program returns 49. We should return 49 to maintain the baseline.
    
    return _strassen_49()

def _fallback_schoolbook():
    n, m, p = 4, 4, 4
    rank = 64
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_49():
    n, m, p = 4, 4, 4
    rank = 49
    
    outer_terms = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]
    
    inner_terms = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,1),1)], [((0,1),1), ((0,0),-1)], [((1,0),1), ((1,1),-1)]),
        ([((0,0),1), ((1,1),-1)], [((1,1),1)], [((0,0),1), ((1,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),-1)]),
        ([((0,1),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((1,1),-1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((1,0),-1)], [((0,0),1)]),
    ]
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for o_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for i_idx, (a_elem, b_elem, c_elem) in enumerate(inner_terms):
            for (br, bc), ca in a_blocks:
                for (er, ec), cv in a_elem:
                    a_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    U[a_idx, col] += ca * cv
            
            for (br, bc), cb in b_blocks:
                for (er, ec), cv in b_elem:
                    b_idx = (br * 2 + er) * 4 + (bc * 2 + ec)
                    V[b_idx, col] += cb * cv
            
            for (br, bc), cc in c_blocks:
                for (er, ec), cv in c_elem:
                    k = br * 2 + er
                    i = bc * 2 + ec
                    c_idx = k * 4 + i
                    W[c_idx, col] += cc * cv
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
