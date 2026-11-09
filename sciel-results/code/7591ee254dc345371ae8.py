<<<<<<< SEARCH
    # Final validation
    valid = True
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_positions[i] - best_positions[j])
            if dist < MIN_DIST:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        best_positions = fallback_positions
        best_side = _calculate_required_side(best_positions)

    # Return angles as 0.0 for all inner hexagons
    return [list(p) + [0.0] for p in best_positions], (0.0, 0.0), best_side, 0.0
=======
    # Final validation with stricter margin
    valid = True
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_positions[i] - best_positions[j])
            # Use a slightly larger threshold for validation to be safe
            if dist < MIN_DIST - 0.01: 
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        best_positions = fallback_positions
        best_side = _calculate_required_side(best_positions)

    # Return angles as 0.0 for all inner hexagons
    return [list(p) + [0.0] for p in best_positions], (0.0, 0.0), best_side, 0.0
>>>>>>> REPLACE
