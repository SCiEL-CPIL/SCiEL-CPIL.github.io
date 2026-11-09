    def search(idx, current_choices):
        nonlocal best_count, best_solution
        
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if len(current_choices) > best_count:
            best_count = len(current_choices)
            best_solution = current_choices[:]
            
        if idx == n_moduli:
            return
            
        # Upper bound
        remaining = n_moduli - idx
        if len(current_choices) + remaining <= best_count:
            return
            
        # ... rest of logic
