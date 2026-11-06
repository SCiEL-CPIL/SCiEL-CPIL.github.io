    sums = [powerful[i] + powerful[j] for i in range(n_p) for j in range(i, n_p)]
    counter = Counter(sums)
    