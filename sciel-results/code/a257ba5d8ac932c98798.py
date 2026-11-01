if delta_e < 0:
    accept = True
else:
    if T > 1e-9:
        accept = rng.random() < np.exp(-delta_e / T)
    else:
        accept = False
