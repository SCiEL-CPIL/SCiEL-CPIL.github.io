<<<<<<< SEARCH
    # 3. If time remains, try more greedy restarts with different weightings
    if time.time() < deadline - 10:
        for w_vec in [weights_uniform, weights_sqrt_inv]:
            if time.time() > deadline - 5:
                break
            alt = fast_greedy(w_vec)
            if len(alt) < best_len:
                best_len = len(alt)
                best_blocks = alt.copy()
                # Quick prune pass
                counts2 = [0] * N_SUBSETS
                for idx in best_blocks:
                    for bit in block_bit_indices[idx]:
                        counts2[bit] += 1
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for pos in range(len(best_blocks)):
                        idx = best_blocks[pos]
                        redundant = True
                        for bit in block_bit_indices[idx]:
                            if counts2[bit] == 1:
                                redundant = False
                                break
                        if redundant:
                            for bit in block_bit_indices[idx]:
                                counts2[bit] -= 1
                            best_blocks.pop(pos)
                            changed = True
                            if len(best_blocks) < best_len:
                                best_len = len(best_blocks)
                            break
=======
    # 3. Final polish: multiple passes of redundant pruning on best_blocks
    if time.time() < deadline - 3:
        counts2 = [0] * N_SUBSETS
        for idx in best_blocks:
            for bit in block_bit_indices[idx]:
                counts2[bit] += 1
        # Multiple aggressive prune passes
        for _pass in range(20):
            if time.time() > deadline - 2:
                break
            changed = False
            for pos in range(len(best_blocks)):
                idx = best_blocks[pos]
                redundant = True
                for bit in block_bit_indices[idx]:
                    if counts2[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    for bit in block_bit_indices[idx]:
                        counts2[bit] -= 1
                    best_blocks.pop(pos)
                    changed = True
                    if len(best_blocks) < best_len:
                        best_len = len(best_blocks)
                    break
            if not changed:
                break
>>>>>>> REPLACE
