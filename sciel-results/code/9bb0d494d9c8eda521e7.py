    size = 0
    # in greedy: size += 1
    # in local search:
    # if add: size += 1
    # if 1-for-2: size += 1 (remove 1, add 2)
    # if kick-refill: new_size = bin(mask_trial).count('1')
    