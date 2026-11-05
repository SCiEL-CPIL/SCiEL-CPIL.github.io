def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Local vars for speed
    seq = sequence
    wh = where
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            x, y = p
            # Check neighbors
            if (x + 1, y) in wh:
                j = wh[(x + 1, y)]
                if j > i + 1 and seq[j] == 'H':
                    contacts += 1
            if (x - 1, y) in wh:
                j = wh[(x - 1, y)]
                if j > i + 1 and seq[j] == 'H':
                    contacts += 1
            if (x, y + 1) in wh:
                j = wh[(x, y + 1)]
                if j > i + 1 and seq[j] == 'H':
                    contacts += 1
            if (x, y - 1) in wh:
                j = wh[(x, y - 1)]
                if j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts
