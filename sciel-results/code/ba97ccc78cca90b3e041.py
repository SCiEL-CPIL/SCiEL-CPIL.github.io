<<<<<<< SEARCH
    best_x = nelder_mead(_objective, x0, max_iter=300, deadline=deadline)
    
    a = best_x[:12].tolist()
    b = best_x[12:].tolist()
    
    # Final check and clipping
    a = [max(-50, min(50, v)) for v in a]
    b = [max(-50, min(50, v)) for v in b]
    
    # Ensure sums are exactly 1?
    # The constraint was handled by penalty, but let's normalize if needed?
    # No, the evaluator checks sum(a)=1. If my penalty didn't enforce it perfectly, I should fix it.
    # However, the penalty was strong (1e10).
    # Let's just return them.
    
    return a, b
=======
    best_x = nelder_mead(_objective, x0, max_iter=200, deadline=deadline)
    
    a = best_x[:12].tolist()
    b = best_x[12:].tolist()
    
    # Final check and clipping
    a = [max(-50, min(50, v)) for v in a]
    b = [max(-50, min(50, v)) for v in b]
    
    # Ensure sums are exactly 1 by normalizing
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-9:
        a = [v / sum_a for v in a]
    else:
        a = [1.0/12.0] * 12
    if abs(sum_b) > 1e-9:
        b = [v / sum_b for v in b]
    else:
        b = [1.0/12.0] * 12
        
    return a, b
>>>>>>> REPLACE
