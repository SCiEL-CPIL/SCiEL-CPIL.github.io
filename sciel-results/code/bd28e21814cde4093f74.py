<<<<<<< SEARCH
    # ---------- Repair (partial greedy on uncovered) ----------
    def repair(uncovered_mask, seed):
        rng = np.random.default_rng(seed)
        unc = uncovered_mask.copy()
        if not np.any(unc):
            return []
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s in np.where(unc)[0]:
            for b in subset_to_blocks[s]:
                gains[b] += 1
        selected = []
        while True:
            if time.time() > deadline - 15.0:
                break
            if gains.max() == 0:
                break
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best = int(np.argmax(scores))
            if gains[best] == 0:
                break
            selected.append(best)
            for s in block_to_subsets[best]:
                if unc[s]:
                    unc[s] = 0
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected
=======
    # ---------- Repair (partial greedy on uncovered) ----------
    def repair(uncovered_mask, seed):
        rng = np.random.default_rng(seed)
        unc = uncovered_mask.copy()
        if not np.any(unc):
            return []
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s in np.where(unc)[0]:
            for b in subset_to_blocks[s]:
                gains[b] += 1
        selected = []
        while True:
            if time.time() > deadline - 15.0:
                break
            if gains.max() == 0:
                break
            # Find all blocks with max gain
            max_gain = gains.max()
            max_blocks = np.where(gains == max_gain)[0]
            if len(max_blocks) > 1:
                best = int(max_blocks[rng.integers(0, len(max_blocks))])
            else:
                best = int(max_blocks[0])
            if gains[best] == 0:
                break
            selected.append(best)
            for s in block_to_subsets[best]:
                if unc[s]:
                    unc[s] = 0
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected
>>>>>>> REPLACE

