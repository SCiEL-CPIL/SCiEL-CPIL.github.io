import math
import numpy as np

def construct():
    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # Efficient computation using NumPy for small to medium N
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        ratio = n_diffs / n_sums
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    def find_golomb_ruler_greedy(n):
        """
        Finds a Golomb ruler of order n using a greedy heuristic.
        This produces a valid Golomb ruler, though not necessarily the shortest.
        """
        marks = [0]
        diffs = set()
        for _ in range(1, n):
            next_mark = marks[-1] + 1
            while True:
                valid = True
                for m in marks:
                    d = next_mark - m
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    marks.append(next_mark)
                    for m in marks[:-1]:
                        diffs.add(next_mark - m)
                    break
                next_mark += 1
        return marks

    # Hardcoded optimal Golomb rulers for N=1 to 10
    rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 28],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47]
    }
    
    best_U = [0]
    best_score = -1
    
    # Evaluate hardcoded optimal rulers
    for N in range(1, 11):
        U = rulers[N]
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Evaluate greedy constructions for larger N
    # The score likely peaks at small N due to the log(max_U) penalty,
    # but we check up to N=20 to be sure.
    for N in range(11, 21):
        U = find_golomb_ruler_greedy(N)
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    return best_U
