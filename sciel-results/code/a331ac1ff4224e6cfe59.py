        t = threading.Thread(
            target=_basin_hopping_worker,
            args=(seed_pos, size, potential, parameter, rng,
                  None, lock, best_energy, best_positions,
                  max_iter=25, bh_scale=0.15)
        )
