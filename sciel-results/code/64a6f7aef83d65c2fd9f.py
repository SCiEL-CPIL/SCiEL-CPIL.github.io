<<<<<<< SEARCH
        # Project onto each normal
        projections = lx * normals[:, 0] + ly * normals[:, 1]  # shape (6, n_verts)
        max_proj = projections.max(axis=1)  # shape (6,)
=======
        # Project onto each normal
        # lx, ly are shape (n_verts,)
        # normals is shape (6, 2)
        # We need shape (6, n_verts)
        projections = lx[None, :] * normals[:, 0:1] + ly[None, :] * normals[:, 1:2]
        max_proj = projections.max(axis=1)  # shape (6,)
>>>>>>> REPLACE
