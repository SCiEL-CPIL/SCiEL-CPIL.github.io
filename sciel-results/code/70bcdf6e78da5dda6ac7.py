        if time.time() > deadline - 8.0:
            rem_indices = [i for i in range(n_edges) if (remaining >> i) & 1]
            order = rem_indices
            sub_color, _ = greedy_color_fast(order, conflicts, n_edges)
            for i in rem_indices:
                color[i] = current_color + sub_color[i]
            current_color += len(set(sub_color[i] for i in rem_indices))
            break
