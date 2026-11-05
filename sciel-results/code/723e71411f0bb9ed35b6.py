def _compute_h_contacts(sequence, chain):
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = set()
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts.add((i, j))
    return contacts
