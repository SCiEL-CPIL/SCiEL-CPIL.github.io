    rem_max = 0
    if i > 0: rem_max = max(rem_max, pfx[i-1])
    if j+1 <= m-1: rem_max = max(rem_max, sfx[j+1])
    if i+1 <= j-1:
        seg_max = 0
        for k in range(i+1, j):
            if edge_pos[k] > seg_max: seg_max = edge_pos[k]
        rem_max = max(rem_max, seg_max)
    