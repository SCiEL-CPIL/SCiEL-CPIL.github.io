        for b_idx in range(len(all_6subsets)):
            subs = block_to_5subsets[b_idx]
            if not any(s in covered for s in subs):
                greedy_blocks.append(list(all_6subsets[b_idx]))
                covered.update(subs)
