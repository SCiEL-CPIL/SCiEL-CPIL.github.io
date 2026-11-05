            pen_U, grad_pen_U = half_int_penalty(U, pen_lam)
            pen_V, grad_pen_V = half_int_penalty(V, pen_lam)
            pen_W, grad_pen_W = half_int_penalty(W, pen_lam)
            total_pen = pen_U + pen_V + pen_W  # This is an array
            
            total_error = tensor_err + total_pen # This becomes an array
