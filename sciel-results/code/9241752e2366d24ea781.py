<<<<<<< SEARCH
    def greedy_coloring(edges, conflicts, order):
        """Greedy coloring in given order."""
        n_edges = len(edges)
        color = [-1] * n_edges

        for e_idx in order:
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c

        return color
=======
    def greedy_coloring(edges, conflicts, order):
        """Greedy coloring in given order."""
        n_edges = len(edges)
        color = [-1] * n_edges
        neighbor_colors_mask = [0] * n_edges

        for e_idx in order:
            # Find smallest available color
            c = 0
            while (neighbor_colors_mask[e_idx] >> c) & 1:
                c += 1
            color[e_idx] = c

            # Update neighbors
            mask_bit = 1 << c
            conf_mask = conflicts[e_idx]
            m = conf_mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if color[j] == -1:
                    neighbor_colors_mask[j] |= mask_bit
                m ^= lsb

        return color
>>>>>>> REPLACE

