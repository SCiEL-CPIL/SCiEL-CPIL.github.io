            dists = np.linalg.norm(candidates[:, np.newaxis, :] - np.array(selected)[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            idx = np.argmax(min_dists)
            selected.append(candidates[idx])
