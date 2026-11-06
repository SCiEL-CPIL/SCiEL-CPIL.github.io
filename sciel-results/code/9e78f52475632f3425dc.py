<<<<<<< SEARCH
    # 3) Fresh builds from a single origin with a few seeds, several radii
    for seed in range(0, 12):
        if time.time() > deadline:
            break
        rng = random.Random(5000 + seed)
        base = [(0, 0)]
        # add a couple of random integer-distance points to seed growth
        for _ in range(3):
            r = rng.randint(10, 60)
            r2 = r * r
            dx = rng.randint(-r, r)
            dy2 = r2 - dx * dx
            dy = math.isqrt(dy2)
            if dy * dy != dy2:
                continue
            cand = (dx, dy) if dy else (dx, 0)
            if cand != (0, 0) and _try_add(base, cand):
                base.append(cand)
        for R in (50, 120):
            if time.time() > deadline:
                break
            p = _greedy_extend(base, R, deadline, seed=6000 + seed * 2 + R)
            maybe_update(p)
=======
    # 3) Fresh builds from a single origin with a few seeds, several radii
    for seed in range(0, 20):
        if time.time() > deadline:
            break
        rng = random.Random(5000 + seed)
        base = [(0, 0)]
        # add a couple of random integer-distance points to seed growth
        for _ in range(4):
            r = rng.randint(5, 80)
            r2 = r * r
            dx = rng.randint(-r, r)
            dy2 = r2 - dx * dx
            dy = math.isqrt(dy2)
            if dy * dy != dy2:
                continue
            cand = (dx, dy) if dy else (dx, 0)
            if cand != (0, 0) and _try_add(base, cand):
                base.append(cand)
        for R in (50, 120, 200):
            if time.time() > deadline:
                break
            p = _greedy_extend(base, R, deadline, seed=6000 + seed * 3 + R)
            maybe_update(p)
>>>>>>> REPLACE

