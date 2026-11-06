import time

def construct():
    N = 16
    # Start with a known 88-comparator network
    comparators = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]
    
    def is_correct(comps):
        for mask in range(1 << N):
            wires = [(mask >> i) & 1 for i in range(N)]
            for i, j in comps:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True

    # Try to remove redundant comparators
    start_time = time.time()
    while len(comparators) > 60 and (time.time() - start_time) < 80:
        improved = False
        for k in range(len(comparators)):
            # Try removing comparator k
            new_comps = comparators[:k] + comparators[k+1:]
            if is_correct(new_comps):
                comparators = new_comps
                improved = True
                break
        if not improved:
            break
            
    return comparators
