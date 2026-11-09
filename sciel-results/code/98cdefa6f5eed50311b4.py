            idxs = np.arange(pos, seg_len, p)
            while True:
                # Get current values
                vals = rem[idxs]
                mask = vals % p == 0
                if not np.any(mask):
                    break
                # Update only those that are divisible
                rem[idxs[mask]] //= p
