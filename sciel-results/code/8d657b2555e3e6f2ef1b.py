<<<<<<< SEARCH
def _local_refine(sequence, chain, rng, max_iter=200, time_budget=10.0):
    """
    Pivot-based local refinement. Randomly pick a residue, try to move it,
    accept if energy improves (or with small probability for exploration).
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy_from_map(sequence, {p: i for i, p in enumerate(chain)}, set(chain))

    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        if time.time() - _T0 > _DEADLINE - time_budget:
            break

        i = rng.randint(1, n - 2)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
=======
def _local_refine(sequence, chain, rng, dimension, max_iter=200, time_budget=10.0):
    """
    Pivot-based local refinement. Randomly pick a residue, try to move it,
    accept if energy improves (or with small probability for exploration).
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy_from_map(sequence, {p: i for i, p in enumerate(chain)}, set(chain))

    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        if time.time() - _T0 > _DEADLINE - time_budget:
            break

        i = rng.randint(1, n - 2)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
>>>>>>> REPLACE

