            c = c / norm_c
            dot_val = np.dot(c, p0)
            if dot_val < 0:
                c = -c
                dot_val = -dot_val
            circumcenters[idx] = c
            radii[idx] = np.arccos(np.clip(dot_val, -1, 1))
