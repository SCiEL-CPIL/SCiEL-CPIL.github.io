        def score(pos):
            # pos: (N, 2)
            # distances between all pairs
            # diff: (N, N, 2)
            # dist: (N, N)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_dist = np.min(dist)
            
            # distance to walls
            # x in [-0.5, 0.5], y in [-0.5, 0.5]
            # margin = min(0.5 - pos[:,0], 0.5 + pos[:,0], 0.5 - pos[:,1], 0.5 + pos[:,1])
            # Actually, min_k (0.5 - |c_ik|) is not quite right for corners?
            # The contract says: min_i min_k (1/2 - |c_ik|)
            # This implies the distance to the closest wall (vertical or horizontal).
            # Wait, if a circle is at (0.4, 0.4), distance to x=0.5 is 0.1, distance to y=0.5 is 0.1.
            # If it's at (0.4, 0.0), distance to x=0.5 is 0.1, distance to y=0.5 is 0.5, y=-0.5 is 0.5.
            # So min_k (0.5 - |c_ik|) is indeed the distance to the closest vertical/horizontal boundary.
            # This is the correct definition for the radius limit imposed by the walls.
            
            wall_dist = np.min(np.minimum(0.5 - np.abs(pos[:, 0]), 0.5 - np.abs(pos[:, 1])), axis=1)
            min_wall_dist = np.min(wall_dist)
            
            return np.min(min_pair_dist / 2.0, min_wall_dist)
        