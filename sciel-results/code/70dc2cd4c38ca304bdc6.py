            if restart < 2:
                # Initialize from known solution, dropping random columns
                # We need to pick 'rank' columns from the 28 known ones
                cols_to_keep = random.sample(range(rank_known), rank)
                
                U_init = np.zeros((8, rank))
                V_init = np.zeros((16, rank))
                W_init = np.zeros((8, rank))
                
                for new_idx, old_idx in enumerate(cols_to_keep):
                    U_init[:, new_idx] = U_known_np[:, old_idx]
                    V_init[:, new_idx] = V_known_np[:, old_idx]
                    W_init[:, new_idx] = W_known_np[:, old_idx]
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random initialization
                x0 = np.random.normal(0, 0.1, total_size)
