# Inside nelder_mead loop, after sorting:
# Check if simplex is too small
if np.std(fvals) < 1e-10:
    break
