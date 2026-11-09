import numpy as np


def kissing_number11() -> np.ndarray:
        d = 11
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        return np.array(points, dtype=np.int64)
