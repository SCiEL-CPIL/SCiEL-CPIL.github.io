"""Direct tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 6]
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Shapes
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Target: Rank 36 (trivial upper bound)
    # Initialize with the trivial decomposition where every (i,j,k) is independent
    R0 = 36
    U0 = np.zeros((dim_u, R0))
    V0 = np.zeros((dim_v, R0))
    W0 = np.zeros((dim_w, R0))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, idx] = 1.0
                V0[j*p + k, idx] = 1.0
                W0[k*n + i, idx] = 1.0
                idx += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0
    
    # Define the tensor equation check using vectorization
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1.0
    
    def check_rank(U, V, W, r):
        # Extract relevant slices
        # U: (n*m, r) -> need U[i*m+j, :]
        # V: (m*p, r) -> need V[j*p+k, :]
        # W: (p*n, r) -> need W[k*n+i, :]
        
        # Reshape U to (n, m, r)
        # Reshape V to (m, p, r)
        # Reshape W to (p, n, r) -> need (p, n, r) but indexed by k*n+i which is (k, i)
        
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute product for all i,j,k,r
        # prod[i, j, k, r] = U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        
        # Efficient way using broadcasting or einsum
        # Let's use einsum: 'ijr, jkr, kir -> ijk' where k in W is the first dim
        # Note: W_t is (p, n, r) so indices are [k, i, r]
        
        try:
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        except Exception:
            return False, 1e10

        err = np.max(np.abs(prod - 1.0))
        return err < 1e-9, err

    def polish_and_check(U, V, W, r, target_err=1e-9):
        """
        Attempt to find a low-rank decomposition.
        Returns (success, U, V, W)
        """
        if time.time() > DEADLINE:
            return False, None, None, None
            
        from scipy.optimize import minimize, differential_evolution
        
        # Strategy 1: Algebraic/Structured Initialization
        # For 2x3x6, the rank is believed to be 6? No, rank of 2x3 is 6? 
        # Actually, for small matrices, exact rank might be higher.
        # Trivial is 36. Can we do 12? (n*p)
        # C[i,k] = sum_j A[i,j] B[j,k].
        # If we fix i and k, we have a scalar mult of sums? No.
        
        # Let's try random restarts with L-BFGS-B from the current best or random starts.
        # Since n,m,p are small, param count is low for rank < 36.
        
        dim_params = r * (dim_u + dim_v + dim_w)
        
        def objective_grad(x):
            U_f = x[:r*dim_u].reshape(dim_u, r)
            V_f = x[r*dim_u:r*dim_u+r*dim_v].reshape(dim_v, r)
            W_f = x[r*dim_u+r*dim_v:].reshape(dim_w, r)
            
            U_t = U_f.reshape(n, m, r)
            V_t = V_f.reshape(m, p, r)
            W_t = W_f.reshape(p, n, r)
            
            # prod[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
            # Using einsum for prod
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            
            # Residual
            res = prod - 1.0
            loss = np.sum(res*res)
            
            # Gradients
            # d loss / d U_t[i,j,r] = 2 * sum_{j',k'} res[i,j',k'] * V_t[j',k',r] * W_t[k',i,r]
            # This is expensive to compute with pure loops, but dims are small.
            # Let's approximate or use autograd if available? No, stick to manual.
            
            U_grad = np.zeros_like(U_t)
            V_grad = np.zeros_like(V_t)
            W_grad = np.zeros_like(W_t)
            
            # Efficient gradient calc using broadcasting
            # gU[i,j,r] = 2 * res[i,j,:] dot (V[:, :, r] * W[:, i, r]) over k?
            # res is (n, m, p)
            # V is (m, p, r)
            # W is (p, n, r)
            
            # For a fixed r:
            # Term in prod for rank r is: U[i,j,r] * V[j,k,r] * W[k,i,r]
            # d prod[i,j,k] / d U[i,j',r] is non-zero only if i==i', j==j'.
            # Actually, let's just loop over i,j,k to build gradients safely since it's small (2*3*6=36 terms).
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val_res = res[i,j,k]
                        for rr in range(r):
                            w_val = W_t[k, i, rr]
                            v_val = V_t[j, k, rr]
                            u_val = U_t[i, j, rr]
                            
                            U_grad[i, j, rr] += 2 * val_res * v_val * w_val
                            V_grad[j, k, rr] += 2 * val_res * u_val * w_val
                            W_grad[k, i, rr] += 2 * val_res * u_val * v_val
                            
            grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
            return loss, grad

        # Initialize parameters
        # Try multiple initializations
        inits = []
        
        # 1. Random small
        rng = np.random.RandomState(42)
        
        # 2. Perturbed trivial (if r <= 36, we can't really perturb trivial easily to get lower rank without dropping cols)
        
        # We will try to find rank r.
        # If r is significantly smaller than 36, random init is key.
        
        # Add some structured inits?
        # For r=6, we might try U as identity-like?
        
        for attempt in range(15):
            if time.time() > DEADLINE:
                break
                
            # Initialize x
            if attempt == 0:
                # Try a structure based on identity if possible
                # Not strictly valid for all, but good start
                U_rand = rng.randn(dim_u, r) * 0.5
                V_rand = rng.randn(dim_v, r) * 0.5
                W_rand = rng.randn(dim_w, r) * 0.5
            else:
                scale = 0.1 * (r ** 0.2)
                U_rand = rng.randn(dim_u, r) * scale
                V_rand = rng.randn(dim_v, r) * scale
                W_rand = rng.randn(dim_w, r) * scale
                
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            # Optimize
            try:
                res_opt = minimize(
                    lambda x: objective_grad(x)[0],
                    x0,
                    jac=lambda x: objective_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6}
                )
            except Exception:
                continue
                
            if res_opt.success or res_opt.fun < 1e-10:
                x_final = res_opt.x
                U_f = x_final[:r*dim_u].reshape(dim_u, r)
                V_f = x_final[r*dim_u:r*dim_u+r*dim_v].reshape(dim_v, r)
                W_f = x_final[r*dim_u+r*dim_v:].reshape(dim_w, r)
                
                # Check validity
                valid, err = check_rank(U_f, V_f, W_f, r)
                if valid:
                    # Attempt to round to halves/ints if close
                    # Try rounding U, V, W to nearest 0.5
                    U_r = np.round(U_f * 2) / 2
                    V_r = np.round(V_f * 2) / 2
                    W_r = np.round(W_f * 2) / 2
                    valid_r, err_r = check_rank(U_r, V_r, W_r, r)
                    
                    if valid_r:
                        return True, U_r, V_r, W_r
                    else:
                        return True, U_f, V_f, W_f
                            
        return False, None, None, None

    # Search for lower ranks
    # Known ranks for small matrices:
    # 1x1x1: 1
    # 2x2x2: 7
    # 2x3x3: ?
    # 2x3x6: ?
    # Trivial is 36.
    # Let's try decreasing from 35 down to 6.
    # Given the small size, we can afford to try a few.
    
    ranks_to_try = list(range(35, 5, -1))
    
    # Prioritize likely lower bounds
    # For n*p = 12, rank >= 12 is a lower bound? 
    # Actually, rank(A*B) for general A,B is at least n*p? No, that's for bilinear complexity which is related but not exactly the same as tensor rank in this specific tensor formulation.
    # The tensor is in V_{nm}^* \otimes V_{mp}^* \otimes V_{pn}^*.
    # Rank >= max(n*m, m*p, p*n) / something? No.
    # Lower bound: rank >= max(n*m, m*p, p*n)? No.
    # Rank >= n*p? 
    # If C = AB, C has n*p entries. Each rank 1 term contributes to one specific C[i,k] via the sum over j? No, a rank 1 term in the tensor decomposition contributes to ALL C[i,k] simultaneously, but sparsely.
    # In the standard matrix multiplication tensor, a rank 1 term is (u_i v_j w_k).
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # If we fix i, the vector of coeffs for j is U[i,j,:].
    # It is known that for n=2, m=3, p=6, the rank is 18? Or 12?
    # Let's just search.
    
    # We have a budget.
    for r in ranks_to_try:
        if time.time() > DEADLINE:
            break
        if r >= best_rank:
            continue
            
        success, U_new, V_new, W_new = polish_and_check(best_U, best_V, best_W, r)
        
        # polish_and_check currently ignores input U,V,W for init, it uses random.
        # Let's modify logic to use best known if possible? No, rank changed.
        
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = r
            # If we find a solution, we might break if we hit theoretical min? 
            # Theoretical min is unknown. We keep searching for lower.
            
    # Final verification
    # Ensure we return valid lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
