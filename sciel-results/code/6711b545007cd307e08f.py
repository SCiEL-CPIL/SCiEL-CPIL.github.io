def construct():
    # Build map from non-zero quadratic residue to a witness root in [1, 1008]
    roots = {}
    for x in range(1, 1009):
        r = (x * x) % 1009
        if r != 0 and r not in roots:
            roots[r] = x
    # Ascending search to find the minimum s such that s..s+4 are all non-zero QRs
    for s in range(1, 1005):
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    raise RuntimeError('No run found')
