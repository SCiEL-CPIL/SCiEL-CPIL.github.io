    # Phase 2b: remove two, re-add greedily
    for it in range(50):
        if time.time() > deadline - 8:
            break
        improved = False
        for x1 in range(1, n+1):
            if not selected[x1]:
                continue
            for x2 in range(x1+1, n+1):
                if not selected[x2]:
                    continue
                if time.time() > deadline - 8:
                    break
                selected[x1] = False
                selected[x2] = False
                added = []
                for x_in in range(1, n+1):
                    if x_in in (x1, x2) or selected[x_in]:
                        continue
                    if can_add(selected, x_in):
                        selected[x_in] = True
                        added.append(x_in)
                if len(added) > 2:
                    best_size = sum(selected[1:])
                    best_sel = selected[:]
                    improved = True
                else:
                    for x_in in added:
                        selected[x_in] = False
                    selected[x1] = True
                    selected[x2] = True
        if not improved:
            break
