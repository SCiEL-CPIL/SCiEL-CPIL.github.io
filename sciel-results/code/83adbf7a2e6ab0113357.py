<<<<<<< SEARCH
    # Initial coarse grid: increased resolution
    depth_min = 0.001
    depth_max = 10.0 / k
    n_depths = 400
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)
=======
    # Initial coarse grid: increased resolution with finer low-depth region
    depth_min = 0.001
    depth_max = 10.0 / k
    n_depths = 500
    # Use a non-uniform grid: more points at low depths where optimal depths concentrate
    # Split into two segments: [depth_min, 5/k] with more points, [5/k, depth_max] with fewer
    split_point = 5.0 / k
    n_low = 300
    n_high = 200
    depth_grid_low = np.geomspace(depth_min, split_point, n_low)
    depth_grid_high = np.geomspace(split_point, depth_max, n_high)
    depth_grid = np.concatenate([depth_grid_low[:-1], depth_grid_high])
>>>>>>> REPLACE
<<<<<<< SEARCH
        # Number of local points: increased for higher precision
        n_local = 100
=======
        # Number of local points: increased for higher precision
        n_local = 120
>>>>>>> REPLACE
<<<<<<< SEARCH
                if qi == n_q - 1:  # q = Q, no bid
                    local_bid_depths = np.array([depth_min])
                    local_ask_depths = np.geomspace(
                        max(prev_ask * 0.8, depth_min),
                        min(prev_ask * 1.25, depth_max),
                        n_local
                    )
                else:
                    local_bid_depths = np.geomspace(
                        max(prev_bid * 0.8, depth_min),
                        min(prev_bid * 1.25, depth_max),
                        n_local
                    )
                    if qi == 0:  # q = -Q, no ask
                        local_ask_depths = np.array([depth_min])
                    else:
                        local_ask_depths = np.geomspace(
                            max(prev_ask * 0.8, depth_min),
                            min(prev_ask * 1.25, depth_max),
                            n_local
                        )
=======
                # Use tighter ratio for refinement passes
                ratio = 1.1 if refine_pass >= 3 else 1.25
                if qi == n_q - 1:  # q = Q, no bid
                    local_bid_depths = np.array([depth_min])
                    local_ask_depths = np.geomspace(
                        max(prev_ask * (2.0 - ratio), depth_min),
                        min(prev_ask * ratio, depth_max),
                        n_local
                    )
                else:
                    local_bid_depths = np.geomspace(
                        max(prev_bid * (2.0 - ratio), depth_min),
                        min(prev_bid * ratio, depth_max),
                        n_local
                    )
                    if qi == 0:  # q = -Q, no ask
                        local_ask_depths = np.array([depth_min])
                    else:
                        local_ask_depths = np.geomspace(
                            max(prev_ask * (2.0 - ratio), depth_min),
                            min(prev_ask * ratio, depth_max),
                            n_local
                        )
>>>>>>> REPLACE
BET: {mechanism: "Finer initial grid concentrated at low depths (where optimal k*d < 5) plus tighter refinement ratio (1.1) in later passes reduces discretization error in the DP value function, yielding more accurate policy lookup", q: 0.4, delta: [0.001, 0.005]}
