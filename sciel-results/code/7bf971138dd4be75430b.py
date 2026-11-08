<<<<<<< SEARCH
    # Fallback to weight-2 vectors if something went wrong
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_set = np.array(base_vectors, dtype=np.int64)

    return best_set
=======
    # Fallback to weight-2 vectors if something went wrong
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_set = np.array(base_vectors, dtype=np.int64)

    return best_set

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to check - the EVOLVE-BLOCK-END is already in the current program. Let me fix that.

