            c, _, _, _ = np.linalg.lstsq(normals, mids, rcond=None)
            return side, c[0], c[1]
    