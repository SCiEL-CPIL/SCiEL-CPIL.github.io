    while time.time() < deadline - 5.0 and attempts < 20:
        attempts += 1
        k = rng.randint(0, 30)
        init = set(tail)
        if k:
            remove_list = rng.sample(list(init), min(k, len(init)))
            for r in remove_list:
                init.discard(r)
        res2 = _local_search(init, deadline, rng)
        if len(res2) > best_size:
            best = res2; best_size = len(res2)
