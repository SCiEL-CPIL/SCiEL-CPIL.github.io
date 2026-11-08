def _make_initial_guess(seed=0):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            r = 0.5 / side * 0.9
            cells.append((cx, cy, r))
    while len(cells) < N:
        cells.append((0.5, 0.5, 0.01))
    
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    return np.concatenate([x, y, r])
