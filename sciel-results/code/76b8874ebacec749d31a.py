<<<<<<< SEARCH
    # Phase 2: deeper annealing from the best + perturbed restarts
    for _ in range(6):
        if remaining() < 8:
            break
        res = local_search(best_rows, rng, 3500, T0=rng.uniform(2.0, 6.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # perturb the best to escape its basin, then re-anneal
            pert = [list(r) for r in best_rows]
            for _ in range(4):
                rr = rng.randrange(n_rows)
                pp = rng.randrange(1, len(pert[rr]))
                lo = pert[rr][pp - 1] + 1
                hi = (pert[rr][pp + 1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 25
                if lo > hi:
                    continue
                nv = rng.randrange(lo, hi + 1)
                new_row = pert[rr][:]
                new_row[pp] = nv
                if not all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                used, ok = set(), True
                for i, row in enumerate(pert):
                    d = nd if i == rr else row_diffs(row)
                    if len(d) != needed or (d & used):
                        ok = False
                        break
                    used |= d
                if ok:
                    pert[rr] = new_row
                    break
            res = local_search(pert, rng, 2500, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res
=======
    # Phase 2: deeper annealing from the best + perturbed restarts + backtracking
    for _ in range(6):
        if remaining() < 8:
            break
        res = local_search(best_rows, rng, 3500, T0=rng.uniform(2.0, 6.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # Try backtracking from a perturbed state
            pert = [list(r) for r in best_rows]
            for _ in range(4):
                rr = rng.randrange(n_rows)
                pp = rng.randrange(1, len(pert[rr]))
                lo = pert[rr][pp - 1] + 1
                hi = (pert[rr][pp + 1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 25
                if lo > hi:
                    continue
                nv = rng.randrange(lo, hi + 1)
                new_row = pert[rr][:]
                new_row[pp] = nv
                if not all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                used, ok = set(), True
                for i, row in enumerate(pert):
                    d = nd if i == rr else row_diffs(row)
                    if len(d) != needed or (d & used):
                        ok = False
                        break
                    used |= d
                if ok:
                    pert[rr] = new_row
                    break
            
            # With some probability, use backtracking to rebuild a row
            if rng.random() < 0.3:
                rr = rng.randrange(n_rows)
                used_other = set()
                for i, r in enumerate(pert):
                    if i != rr:
                        used_other |= row_diffs(r)
                row = [0]
                max_branch = rng.choice([4, 5, 6])
                
                def bt_row(row, used, depth=0):
                    if len(row) == k + 1:
                        return True
                    if remaining() < 5:
                        return False
                    cands = []
                    y = row[-1] + 1
                    scanned = 0
                    while len(cands) < max_branch and scanned < 300:
                        scanned += 1
                        new = {y - a for a in row}
                        if not (new & used):
                            cands.append(y)
                        y += 1
                    if not cands:
                        while True:
                            new = {y - a for a in row}
                            if not (new & used):
                                cands.append(y)
                                break
                            y += 1
                    rng.shuffle(cands)
                    for cand in cands:
                        new_used = used | {cand - a for a in row}
                        if bt_row(row + [cand], new_used, depth + 1):
                            return True
                    return False
                
                if bt_row(row, used_other):
                    pert[rr] = row
            
            res = local_search(pert, rng, 2500, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res
>>>>>>> REPLACE
