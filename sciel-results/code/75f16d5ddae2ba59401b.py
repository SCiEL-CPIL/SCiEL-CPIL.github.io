# Old:
mask = (cover_count[neighbors_list[c]] == 1)
unique_words = neighbors_list[c][mask]

# New:
nc = np.array(neighbors_list[c])
mask = (cover_count[nc] == 1)
unique_words = nc[mask]
