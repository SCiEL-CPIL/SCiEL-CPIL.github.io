uncovered = [1] * 6561
code = []
while any(uncovered):
    best_idx = -1
    best_count = -1
    for j in range(6561):
        count = 0
        for k in neighbors[j]:
            if uncovered[k]:
                count += 1
        if count > best_count:
            best_count = count
            best_idx = j
            if count == 17: # Max possible
                break # Can't do better than 17
    code.append(best_idx)
    for k in neighbors[best_idx]:
        uncovered[k] = 0
