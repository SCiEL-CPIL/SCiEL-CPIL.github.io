    mask = (np.arange(dim) >> qubit) & 1
    idx0 = np.where(mask == 0)[0]
    idx1 = np.where(mask == 1)[0]
    # ...
    