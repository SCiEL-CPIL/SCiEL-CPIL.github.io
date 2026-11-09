<<<<<<< SEARCH
    def angles_from_params(x):
        # x are log-weights
        w = np.exp(x - np.max(x)) # for numerical stability
        sum_w = np.sum(w)
        gaps = w / sum_w * 2 * np.pi
        # First angle is 0 (rotation invariance)
        angles = np.cumsum(np.concatenate(([0.0], gaps)))[:-1]
        # angles[0] = 0
        # angles[1] = gap[0]
        # ...
        # angles[18] = sum(gaps[0..17])
        # The last gap is implicit in the wrap-around.
        return angles
=======
    def angles_from_params(x):
        # x are log-weights
        w = np.exp(x - np.max(x)) # for numerical stability
        sum_w = np.sum(w)
        gaps = w / sum_w * 2 * np.pi
        # First angle is 0 (rotation invariance)
        # We need 20 angles for 20 points.
        # gaps has 19 elements.
        # concat([0], gaps) has 20 elements.
        # cumsum has 20 elements.
        # These correspond to angles of P0...P19.
        angles = np.cumsum(np.concatenate(([0.0], gaps)))
        return angles
>>>>>>> REPLACE
