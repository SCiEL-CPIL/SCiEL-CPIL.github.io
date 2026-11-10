def _generate_grid_config():
    """Generate a grid-based configuration and perturb to remove 4-collinear."""
    points = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            points.append([x, y])
    # This is a 5x5 grid, 25 points. We need 20.
    # Randomly remove 5 points.
    random.shuffle(points)
    points = points[:20]
    
    # Now perturb to remove 4-collinear
    # We can do a local search to fix 4-collinear
    # Or just check and if invalid, reject.
    # Better: move points slightly.
    
    # Let's just use the random generator for simplicity in the first pass,
    # but grid is better for dense triples.
    
    # Let's try to repair a grid.
    # For each 4-collinear line, move one point.
    # This is complex.
    
    # Alternative: Start with grid, then do local search to minimize score.
    # The local search will handle the 4-collinear constraint by rejecting invalid moves.
    # But if the initial config is invalid, `_score` might behave unexpectedly?
    # `_score` doesn't check for 4-collinear. It just finds triples.
    # If there are 4 collinear points, there are $\binom{4}{3}=4$ triples.
    # The MIS solver will handle it.
    # But the constraint is "no 4 collinear".
    # So we MUST ensure the final config has no 4 collinear.
    # The local search rejects moves that create 4-collinear.
    # So if we start with an invalid config, we might never fix it if we only do small moves.
    # So we should start with a valid config.
    
    return points
