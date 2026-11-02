<<<<<<< SEARCH
    # Precompute the rule for the LUW construction
    # The rule maps c to (a, b) for the equation y[c] = x[c] + y[a]*x[b] (or similar)
    # The seed code has a specific rule generation.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
=======
    # Precompute the rule for the LUW construction
    # Use a "chain" rule to maximize girth: y[c] depends on y[c-1]
    rule = {}
    for c in range(2, d + 1):
        rule[c] = (c - 1, 1)
>>>>>>> REPLACE
