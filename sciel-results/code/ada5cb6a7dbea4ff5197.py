<<<<<<< SEARCH
    # Extract best result
    x_best = result.x
    centers_best = x_best[:2*n].reshape(n, 2)
    sides_best = x_best[2*n:]
    
    # Ensure validity
    # Clip sides to positive
    sides_best = np.maximum(sides_best, 1e-6)
    
    # Final check and cleanup
    # If the optimization resulted in a bad penalty, we might have overlaps.
    # We can try to "fix" by reducing sides slightly if there are overlaps.
    # But L-BFGS-B with strong penalty should handle it.
    
    return centers_best, sides_best
=======
    # Extract best result
    x_best = result.x
    centers_best = x_best[:2*n].reshape(n, 2)
    sides_best = x_best[2*n:]
    
    # Ensure validity
    # Clip sides to positive
    sides_best = np.maximum(sides_best, 1e-6)
    
    # Check if the result is valid
    if is_valid(centers_best, sides_best):
        return centers_best, sides_best
    else:
        # Fallback to initial valid configuration
        return centers, sides
>>>>>>> REPLACE
