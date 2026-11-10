        for sp in shape_points:
            # sp is shape (3, 5)
            x = np.concatenate([sp, pos_mid])
