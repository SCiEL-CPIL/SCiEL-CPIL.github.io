def _refine(positions, potential):
    try:
        energy, relaxed = relax(positions, potential)
        return energy, relaxed
    except Exception:
        return np.inf, positions
