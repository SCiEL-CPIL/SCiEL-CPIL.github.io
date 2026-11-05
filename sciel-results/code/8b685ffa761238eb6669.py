<<<<<<< SEARCH
    def deletion_pass(selected):
        """Remove redundant blocks."""
        if not selected:
            return selected
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_sel = []
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                essential = False
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential = True
                        break
                if essential:
                    new_sel.append(b_idx)
                else:
                    changed = True
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            selected = new_sel
        return selected
=======
    def deletion_pass(selected):
        """Remove redundant blocks."""
        if not selected:
            return selected
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_t_idxs[b_idx]] += 1
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_sel = []
            for b_idx in selected:
                b_t_idxs = block_t_idxs[b_idx]
                if np.any(cover_count[b_t_idxs] == 1):
                    new_sel.append(b_idx)
                else:
                    changed = True
                    cover_count[b_t_idxs] -= 1
            selected = new_sel
        return selected
>>>>>>> REPLACE

