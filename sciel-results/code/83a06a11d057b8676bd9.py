    x_vals = np.arange(1, p)
    residues = (x_vals * x_vals) % p
    # ...
    for x, r in zip(valid_x, valid_residues):
        if root_of_residue[r] == -1:
            root_of_residue[r] = x
