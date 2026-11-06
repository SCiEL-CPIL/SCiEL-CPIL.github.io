def ensure_fast(v):
    if v in fast:
        return
    if v in slow:
        slow.remove(v)
        fast.add(v)
        ops.append(['load', v])
        return
    # absent: recompute
    # v must not be an input (inputs are in slow initially)
    for p in parents.get(v, []):
        ensure_fast(p)
    fast.add(v)
    ops.append(['compute', v])
