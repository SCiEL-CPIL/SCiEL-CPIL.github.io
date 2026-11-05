            if i > 0:
                left_term = overlap[current_seq[i-1], seq_j]
            else:
                left_term = np.zeros(len(j_indices), dtype=np.int32)
