# Inside _als_decompose, before the for a loop:
design_U = np.einsum('br,cr->rbc', V, W).reshape(R, -1)
# Then inside loop:
U[a, :] = np.linalg.lstsq(design_U, target, rcond=None)[0]
