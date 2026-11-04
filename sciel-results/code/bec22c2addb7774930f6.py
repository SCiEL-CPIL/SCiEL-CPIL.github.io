    axes_all = get_axes(A) # (N, 6, 2)
    # ...
    axes_i = axes_all[i]
    axes_j = axes_all[j]
    axes = np.vstack([axes_i, axes_j])
    # Project all_verts[i] and all_verts[j] onto axes
