"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    roots = {}
    for x in range(1, 1009):
        r = (x * x) % 1009
        if r != 0 and r not in roots:
            roots[r] = x
    # Ascending search to find the minimum s
    for s in range(1, 1005):
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
