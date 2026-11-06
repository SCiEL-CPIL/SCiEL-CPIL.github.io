wall_dist = np.min(np.minimum(0.5 - np.abs(centers), 
                               np.stack([0.5 - np.abs(centers), 0.5 - np.abs(centers), 0.5 - np.abs(centers)], axis=-1)), axis=-1)
