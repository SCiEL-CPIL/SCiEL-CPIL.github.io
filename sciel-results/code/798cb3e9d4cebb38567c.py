            candidates = []
            for e in range(num_edges_total):
                if removed_edges[e]:
                    continue
                if counts[e] > max_count:
                    max_count = counts[e]
                    best_edge = e
                    candidates = [e]
                elif counts[e] == max_count and max_count > 0:
                    candidates.append(e)
