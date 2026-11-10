    seq = []
    i = best_i
    j = best_j
    while True:
        seq.append(primes[i])
        if j == -1:
            break
        i, j = j, prev[i][j]
    
    seq.reverse()
    return seq
