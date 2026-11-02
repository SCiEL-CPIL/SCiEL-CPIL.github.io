   def solve_m22p(p, R, n_restarts=50, max_iter=1000):
       # T shape: (4, 2*p, 2*p)
       T = np.zeros((4, 2*p, 2*p))
       # Fill T
       for i in range(2):
           for j in range(2):
               for k in range(p):
                   # A[i,j] * B[j, 2*k] -> C[i, 2*k]
                   # A[i,j] * B[j, 2*k+1] -> C[i, 2*k+1]
                   # Wait, B is 2 x p.
                   # B row j, col k.
                   # C[i, k] = sum_j A[i,j] B[j,k]
                   # Indices:
                   # A index: i*2 + j
                   # B index: j*p + k
                   # C index: i*p + k
                   a_idx = i*2 + j
                   b_idx = j*p + k
                   c_idx = i*p + k
                   T[a_idx, b_idx, c_idx] = 1
                   
       # Flatten T
       T_flat = T.flatten()
       
       def model(params):
           # params: U(4*R), V(2p*R), W(2p*R)
           U = params[:4*R].reshape(4, R)
           V = params[4*R:4*R+2*p*R].reshape(2*p, R)
           W = params[4*R+2*p*R:].reshape(2*p, R)
           
           # Compute tensor sum
           # result[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
           # Use einsum
           res = np.einsum('ir,br,cr->ibc', U, V, W)
           return res.flatten()
       
       best_res = np.inf
       best_params = None
       
       for _ in range(n_restarts):
           # Random init
           # Scale: entries around 1
           init = np.random.randn(4*R + 4*p*R) * 0.5
           # Try to satisfy
           try:
               res = scipy.optimize.least_squares(model, init, method='lm', max_nfev=max_iter)
               if res.cost < best_res:
                   best_res = res.cost
                   best_params = res.x
           except:
               pass
               
       if best_res < 1e-8:
           # Reconstruct U, V, W
           U = best_params[:4*R].reshape(4, R)
           V = best_params[4*R:4*R+2*p*R].reshape(2*p, R)
           W = best_params[4*R+2*p*R:].reshape(2*p, R)
           # Convert to list of lists of floats/ints
           # Check if they are close to integers/half-integers
           # If not, return None
           return U, V, W
       else:
           return None
   