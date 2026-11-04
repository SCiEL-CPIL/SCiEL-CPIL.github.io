import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_t = np.zeros((n * m, rank_trivial))
    V_t = np.zeros((m * p, rank_trivial))
    W_t = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t[i * m + j, col] = 1
                V_t[j * p + k, col] = 1
                W_t[k * n + i, col] = 1
                col += 1
    
    # Target Tensor T flattened
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    T_flat = T.flatten()

    # Try to find lower rank
    for r in range(rank_trivial - 1, 0, -1):
        # Define parameter vector
        # Size: r * (nm + mp + pn)
        size = r * (n * m + m * p + p * n)
        
        def residual(params):
            U = params[0 : r * n * m].reshape(r, n * m).T
            V = params[r * n * m : r * (n * m + m * p)].reshape(r, m * p).T
            W = params[r * (n * m + m * p) :].reshape(r, p * n).T
            
            # Compute tensor sum
            # Sum over l of U[:, l] outer V[:, l] outer W[:, l]
            # This is expensive if done naively.
            # Use einsum
            # T_comp[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            T_comp = np.einsum('al,bl,cl->abc', U, V, W)
            return (T_comp - T).flatten()

        # Initial guess: Random small numbers
        best_res = np.inf
        for attempt in range(10):
            x0 = np.random.randn(size) * 0.1
            try:
                res = least_squares(residual, x0, method='lm', max_nfev=1000)
                if res.cost < best_res:
                    best_res = res.cost
                    best_x = res.x
            except:
                pass
        
        if best_res < 1e-10:
            # Extract U, V, W
            U = best_x[0 : r * n * m].reshape(r, n * m).T
            V = best_x[r * n * m : r * (n * m + m * p)].reshape(r, m * p).T
            W = best_x[r * (n * m + m * p) :].reshape(r, p * n).T
            
            # Check if close to half-integers
            # Round to nearest 0.5
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            T_comp_int = np.einsum('al,bl,cl->abc', U_int, V_int, W_int)
            if np.allclose(T_comp_int, T, atol=1e-9):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            else:
                # If not close to half-integers, we might fail the constraint.
                # But the prompt says "integer/half-integer entries".
                # If we can't guarantee it, we should probably return the trivial one.
                # However, often these problems have integer solutions.
                # Let's check if U, V, W are close to integers.
                U_int2 = np.round(U)
                V_int2 = np.round(V)
                W_int2 = np.round(W)
                T_comp_int2 = np.einsum('al,bl,cl->abc', U_int2, V_int2, W_int2)
                if np.allclose(T_comp_int2, T, atol=1e-9):
                    return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
                # Otherwise, discard this r and try lower? Or keep best?
                # If we can't satisfy the integer constraint, we should be conservative.
                # But maybe the floating point solution IS the integer solution with noise.
                # If the residual is 0, the values are exact.
                # If they are not half-integers, the evaluator might reject it.
                # Let's assume if residual is 0, we can return the floats?
                # "with integer/half-integer entries" is a requirement.
                # So we MUST return half-integers.
                # If we can't find a half-integer solution, we should return trivial.
                pass

    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
