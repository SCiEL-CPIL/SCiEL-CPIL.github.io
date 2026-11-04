import numpy as np
import time
from scipy.optimize import least_squares

def construct():
       n, m, p = 3, 3, 5
       nm, mp, pn = n*m, m*p, p*n
       schoolbook_rank = nm * p # 45
       
       def residuals(params):
           # params length: R * (nm + mp + pn)
           # U: first nm*R
           # V: next mp*R
           # W: last pn*R
           
           R = len(params) // (nm + mp + pn)
           if R == 0: return []
           
           U_flat = params[:nm*R].reshape(nm, R)
           V_flat = params[nm*R:nm*R+mp*R].reshape(mp, R)
           W_flat = params[nm*R+mp*R:].reshape(pn, R)
           
           res = np.zeros(n*p) # 15 values? No, 45 values.
           # Residual for each (i, j, k)
           # The tensor has 45 components.
           # res_idx = 0
           # for i in range(n):
           #   for j in range(m):
           #     for k in range(p):
           #       val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
           #       res[res_idx] = val - 1.0
           #       res_idx += 1
           # return res
           
           # Vectorized:
           # This is hard to fully vectorize without broadcasting to (n, m, p, R)
           # n=3, m=3, p=5, R=40.
           # Shape (3, 3, 5, 40) is small.
           
           # U[i, j, c] -> U_flat[i*m+j, c]
           # V[j, k, c] -> V_flat[j*p+k, c]
           # W[k, i, c] -> W_flat[k*n+i, c]
           
           # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c]
           
           # Let's just use loops, it's fast enough for 45 terms.
           res = []
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       u = U_flat[i*m+j]
                       v = V_flat[j*p+k]
                       w = W_flat[k*n+i]
                       val = np.dot(u, v * w)
                       res.append(val - 1.0)
           return np.array(res)

       best_U, best_V, best_W = None, None, None
       best_rank = schoolbook_rank
       
       # Try to find lower ranks
       for target_rank in range(schoolbook_rank - 1, 30, -1):
           # Stop if time is running out
           if time.time() > 100: break
           
           # Multiple restarts
           for restart in range(5):
               # Initialize with small random values
               rng = np.random.default_rng(restart)
               U_init = rng.normal(size=(nm, target_rank))
               V_init = rng.normal(size=(mp, target_rank))
               W_init = rng.normal(size=(pn, target_rank))
               
               # Scale to avoid overflow/underflow
               U_init *= 0.1
               V_init *= 0.1
               W_init *= 0.1
               
               x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
               
               try:
                   res = least_squares(residuals, x0, method='lm', x_scale='jac')
                   if res.cost < 1e-8:
                       # Success!
                       U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                       V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                       W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                       
                       # Convert to list of lists of ints/half-ints?
                       # The problem asks for integer/half-integer entries.
                       # The optimizer might produce floats.
                       # We need to round them?
                       # If they are not close to 0.5 or 1.0, it's invalid.
                       
                       # Check if entries are close to 0, 0.5, 1, -1, etc.
                       # Actually, the problem says "integer/half-integer".
                       # This implies the solution space is discrete?
                       # Or just that the *optimal* solution has such entries?
                       # "Find a rank-minimal exact tensor decomposition ... with integer/half-integer entries"
                       # This is a strong constraint.
                       # Random gradient descent will likely find a float solution.
                       # Rounding might break the decomposition.
                       
                       # So I should only accept if the rounded values still satisfy the equation.
                       
                       U_rounded = np.round(U_sol * 2) / 2
                       V_rounded = np.round(V_sol * 2) / 2
                       W_rounded = np.round(W_sol * 2) / 2
                       
                       # Verify rounded solution
                       def verify_rounded(U, V, W, R):
                           for i in range(n):
                               for j in range(m):
                                   for k in range(p):
                                       u = U[i*m+j]
                                       v = V[j*p+k]
                                       w = W[k*n+i]
                                       val = np.dot(u, v * w)
                                       if abs(val - 1.0) > 1e-6:
                                           return False
                           return True
                       
                       if verify_rounded(U_rounded, V_rounded, W_rounded, target_rank):
                           best_U = U_rounded.tolist()
                           best_V = V_rounded.tolist()
                           best_W = W_rounded.tolist()
                           best_rank = target_rank
                           break # Found a lower rank, try lower still? Or break?
                           # Let's try to find even lower.
                   
               except Exception:
                   pass
           
           if best_rank < schoolbook_rank:
               # We found a valid lower rank. Can we go lower?
               # The loop continues to try lower ranks.
               pass

       if best_U is None:
           # Return schoolbook
           rank = schoolbook_rank
           U = [[0] * rank for _ in range(nm)]
           V = [[0] * rank for _ in range(mp)]
           W = [[0] * rank for _ in range(pn)]
           idx = 0
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       U[i*m+j][idx] = 1
                       V[j*p+k][idx] = 1
                       W[k*n+i][idx] = 1
                       idx += 1
           best_U, best_V, best_W = U, V, W

       return {"U": best_U, "V": best_V, "W": best_W}
