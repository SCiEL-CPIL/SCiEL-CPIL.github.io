<<<<<<< SEARCH
def _initial_candidates():
    cands = []
    # evenly spaced
    cands.append([0.5 + 1.5 * i for i in range(12)])
    cands.append([1.0 + 1.0 * i for i in range(12)])
    cands.append([0.5 + 0.7 * i for i in range(12)])
    # geometric
    cands.append([0.5 * (1.2 ** i) for i in range(12)])
    cands.append([0.3 * (1.3 ** i) for i in range(12)])
    # small cluster + tail
    cands.append([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5, 10.0])
    return cands
=======
def _initial_candidates():
    cands = []
    # evenly spaced
    cands.append([0.5 + 1.5 * i for i in range(12)])
    cands.append([1.0 + 1.0 * i for i in range(12)])
    cands.append([0.5 + 0.7 * i for i in range(12)])
    cands.append([0.3 + 0.5 * i for i in range(12)])
    # geometric
    cands.append([0.5 * (1.2 ** i) for i in range(12)])
    cands.append([0.3 * (1.3 ** i) for i in range(12)])
    cands.append([0.4 * (1.25 ** i) for i in range(12)])
    # small cluster + tail
    cands.append([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5, 10.0])
    cands.append([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0])
    # clustered at low values
    cands.append([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0])
    return cands
>>>>>>> REPLACE

