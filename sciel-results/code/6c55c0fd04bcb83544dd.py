def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = [sequences[label] for label in labels]
    n_sites = len(seqs[0])
    alignment = np.zeros((n, n_sites), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        for j, ch in enumerate(seq):
            alignment[i, j] = char_to_int.get(ch, 0)
    
    # Broadcasting: diff[i,j,k] = 1 if alignment[i,k] != alignment[j,k]
    # diff shape: (n, n, n_sites)
    diff = (alignment[:, None, :] != alignment[None, :, :])
    counts = diff.sum(axis=2)  # shape (n, n)
    p = counts / n_sites
    
    D = np.where(p >= 0.7499, 3.0, -0.75 * np.log(1.0 - 4.0 * p / 3.0))
    np.fill_diagonal(D, 0.0)
    return D
