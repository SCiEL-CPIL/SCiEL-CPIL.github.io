def local_search_best(c):
    c = c.copy()
    current_norm = eval_norm(c)
    improved = True
    while improved:
        improved = False
        best_i = -1
        best_new_norm = current_norm
        
        for i in range(N_COEFFS):
            c[i] *= -1
            new_norm = eval_norm(c)
            if new_norm < best_new_norm:
                best_new_norm = new_norm
                best_i = i
            c[i] *= -1 # Revert for next trial
            
        if best_i != -1:
            c[best_i] *= -1 # Apply the best move
            current_norm = best_new_norm
            improved = True
            
    return c
