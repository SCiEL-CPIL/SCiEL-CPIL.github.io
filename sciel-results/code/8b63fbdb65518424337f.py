    for seed in [42, 43, 44]:
        result = _find_valid_coloring(17, time_limit=10.0, seed=seed)
        if result is not None:
            return {'n': 17, 'colors': result}
