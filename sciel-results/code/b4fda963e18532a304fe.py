    for s in range(1, 1005):
        if (is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]):
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
