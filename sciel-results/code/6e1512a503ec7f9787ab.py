    best_colors = colors.copy()
    best_valid = (bad_count == 0)
    
    # In fix loop, if we find a valid state, update best.
    # Actually, we only change colors if valid. So current state is always valid?
    # No, we revert if not valid. So current state is always valid (if we start valid).
    # But we start with 1-factorization which might be invalid.
    # So we need to track validity.
