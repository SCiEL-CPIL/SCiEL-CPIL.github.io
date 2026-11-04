if need == 0:
    cand = tuple(sorted(base))
    cidx = block_index[cand]
    if cidx in rem:
        # bk already covers R; just remove bi, bj
        cov[block_triples[bi]] -= 1
        cov[block_triples[bj]] -= 1
        del chosen[b]
        del chosen[a]
        improved = True; done=True; break
    else:
        newblk = cand
        ... add ...
