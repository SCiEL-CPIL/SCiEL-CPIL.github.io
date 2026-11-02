<<<<<<< SEARCH
    # Step 3: If time remains, try randomised greedy with local improvement
    if time.time() < deadline - 10.0:
        # Try several randomised greedy constructions
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
            rng2 = random.Random(seed)
            used, rows = set(), []
            success = True
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Try next few candidates, pick one that minimizes future conflict
                    found = False
                    window = 1 + (seed % 7)
                    candidates = []
                    for cand in range(x + 1, x + 1 + window):
                        new = {cand - a for a in row}
                        if not new & used:
                            candidates.append((sum(new), cand))
                    if candidates:
                        candidates.sort()
                        if rng2.random() < 0.7:
                            chosen = candidates[0][1]
                        else:
                            chosen = candidates[rng2.randrange(len(candidates))][1]
                        row.append(chosen)
                        used |= {chosen - a for a in row[:-1]}
                        x = chosen
                        found = True
                    else:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            found = True
                        else:
                            continue
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                sc = _scope(rows)
                if sc < best_M:
                    best_rows = rows
                    best_M = sc
=======
    # Step 3: If time remains, try randomised greedy with local improvement
    if time.time() < deadline - 10.0:
        # Try several randomised greedy constructions
        for seed in range(30):
            if time.time() > deadline - 5.0:
                break
            rng2 = random.Random(seed)
            used, rows = set(), []
            success = True
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Try next few candidates, pick one that minimizes future conflict
                    found = False
                    window = 1 + (seed % 11)
                    candidates = []
                    for cand in range(x + 1, x + 1 + window):
                        new = {cand - a for a in row}
                        if not new & used:
                            # Score: prefer candidates that use "less dense" differences
                            candidates.append((sum(new), cand))
                    if candidates:
                        candidates.sort()
                        if rng2.random() < 0.6:
                            chosen = candidates[0][1]
                        else:
                            chosen = candidates[rng2.randrange(len(candidates))][1]
                        row.append(chosen)
                        used |= {chosen - a for a in row[:-1]}
                        x = chosen
                        found = True
                    else:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            found = True
                        else:
                            continue
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                sc = _scope(rows)
                if sc < best_M:
                    best_rows = rows
                    best_M = sc
>>>>>>> REPLACE
