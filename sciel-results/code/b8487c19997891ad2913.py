        chunk_size = 2000
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            chunk = pw_arr[i_start:i_end]
            # sums_matrix: (chunk_size, n)
            sums_matrix = chunk[:, None] + pw_arr[None, :]
            # mask: (chunk_size, n)
            # We want b >= a, so pw_arr >= chunk
            mask = pw_arr[None, :] >= chunk[:, None]
            valid_sums = sums_matrix[mask]
            if len(valid_sums) > 0:
                vals, counts = np.unique(valid_sums, return_counts=True)
                for v, c in zip(vals, counts):
                    sum_counts[v] = sum_counts.get(v, 0) + c
            
            if time.time() > deadline - 10:
                break
