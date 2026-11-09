    for i, k in enumerate(kinds):
        rng = np.random.default_rng(base_seed + i * 100)
        pos = _generate_initial_structures(n, rng, [k])[0]
        structures.append(pos)
