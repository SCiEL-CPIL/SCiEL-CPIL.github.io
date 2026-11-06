def max_sidon_size(arr):
    n = len(arr)
    best = 0
    # Check sizes from n down to best+1
    for k in range(n, best, -1):
        # Generate combinations of size k
        # If any is Sidon, best = k and break
        found = False
        for combo in combinations(arr, k):
            if is_sidon(combo):
                best = k
                found = True
                break
        if found:
            break
    return best
