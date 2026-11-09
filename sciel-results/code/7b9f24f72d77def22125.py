    neighbor_lists = []
    for i in range(total_words):
        neighbor_lists.append(list(np.where(covers[i])[0]))
