def construct():
    deadline = time.time() + 140.0
    
    # ... (helper functions from before, updated with new constraint) ...
    
    def constraint_fun(x, i, j):
        xi = x[i*3]
        yi = x[i*3+1]
        ai = x[i*3+2]
        xj = x[j*3]
        yj = x[j*3+1]
        aj = x[j*3+2]
        
        dx = xi - xj
        dy = yi - yj
        d2 = dx*dx + dy*dy
        
        alpha_rel = abs(ai - aj)
        d_req = get_min_dist(alpha_rel)
        
        return d2 - d_req*d_req

    # ... (rest of optimization) ...
