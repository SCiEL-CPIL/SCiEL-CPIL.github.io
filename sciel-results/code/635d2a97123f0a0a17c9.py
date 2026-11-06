    colors = [bin(i).count('1') % 2 for i in range(n)]
    is_mono = [False] * total_aps
    # ...
    for p, (i1, i2, i3, i4) in enumerate(aps):
        if colors[i1] == colors[i2] and colors[i2] == colors[i3] and colors[i3] == colors[i4]:
            is_mono[p] = True
            total_mono += 1
    # ...
    def calc_flip(i, colors, is_mono):
        c_new = 1 - colors[i]
        delta = 0
        affected = []
        new_statuses = []
        
        for p in ap_indices_per_bit[i]:
            if is_mono[p]:
                delta -= 1
                affected.append(p)
                new_statuses.append(False)
            else:
                i1, i2, i3, i4 = aps[p]
                if i == i1:
                    if colors[i2] == c_new and colors[i3] == c_new and colors[i4] == c_new:
                        delta += 1
                        affected.append(p)
                        new_statuses.append(True)
                    else:
                        affected.append(p)
                        new_statuses.append(False)
                elif i == i2:
                    if colors[i1] == c_new and colors[i3] == c_new and colors[i4] == c_new:
                        delta += 1
                        affected.append(p)
                        new_statuses.append(True)
                    else:
                        affected.append(p)
                        new_statuses.append(False)
                elif i == i3:
                    if colors[i1] == c_new and colors[i2] == c_new and colors[i4] == c_new:
                        delta += 1
                        affected.append(p)
                        new_statuses.append(True)
                    else:
                        affected.append(p)
                        new_statuses.append(False)
                else: # i == i4
                    if colors[i1] == c_new and colors[i2] == c_new and colors[i3] == c_new:
                        delta += 1
                        affected.append(p)
                        new_statuses.append(True)
                    else:
                        affected.append(p)
                        new_statuses.append(False)
                        
        return delta, affected, new_statuses
