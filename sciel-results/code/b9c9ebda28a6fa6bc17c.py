<<<<<<< SEARCH
            # Convert center projections back to x,y: we need to solve
            # center @ normals.T = center_proj
            # This is a 2x6 system, solve least squares
            center, _, _, _ = np.linalg.lstsq(normals.T, center_proj, rcond=None)
=======
            # Convert center projections back to x,y: we need to solve
            # normals @ center = center_proj
            # This is a 6x2 system, solve least squares
            center, _, _, _ = np.linalg.lstsq(normals, center_proj, rcond=None)
>>>>>>> REPLACE
