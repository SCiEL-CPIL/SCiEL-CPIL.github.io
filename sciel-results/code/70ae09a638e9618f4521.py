    import numpy as np
    import itertools

    def construct():
        D = 11
        vectors = []
        # Type 3: exactly three non-zero entries, each +/- 1
        for positions in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                vectors.append(v)
        
        return np.array(vectors, dtype=np.int64)
    