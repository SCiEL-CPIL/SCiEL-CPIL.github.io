    for idx, v in enumerate(topo):
        if time.time() > deadline:
            break
        # ensure parents in fast
        for p in parents[v]:
            ensure(p, idx)
        # ensure v in fast
        if v not in fast:
            if v in inputs:
                # load the input
                if v in in_slow:
                    ops.append(['load', v])
                    transfers[0] += 1
                    fast.add(v)
                    evict(idx, set())
                else:
                    # shouldn't happen; recompute not possible for input
                    pass
            else:
                ops.append(['compute', v])
                fast.add(v)
                evict(idx, set())
