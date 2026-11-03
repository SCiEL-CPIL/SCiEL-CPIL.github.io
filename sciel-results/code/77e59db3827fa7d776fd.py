from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 120.0
    
    x = np.linspace(-0.25, 0.25, N)
    h0 = np.exp(-((x/0.25)**2) * 5.0)
    
    def obj(h):
        val, _ = score_and_grad(h)
        return val
    
    def grad(h):
        _, g = score_and_grad(h)
        return g
    
    bounds = [(0, None)] * N
    
    result = minimize(obj, h0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'gtol': 1e-6, 'maxls': 50})
    
    if time.time() - start_time > deadline - 10:
        return list(h0)
    
    return list(result.x)
