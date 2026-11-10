import time
import numpy as np

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 130
    divisors = get_divisors(N)

    def greedy(order):
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() > deadline - 15:
                break
            best_r = 0
            max_cov = 0
            for r in range(m):
                idx = np.arange(r, N, m)
                cov = int(np.sum(covered[idx] == 0))
                if cov > max_cov:
                    max_cov = cov
                    best_r = r
            if max_cov > 0:
                classes.append([m, best_r])
                covered[np.arange(best_r, N, m)] = 1
        return classes, int(np.sum(covered == 0))

    def improve(classes):
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[np.arange(r, N, m)] = 1
        cur = int(np.sum(covered == 0))
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, (m, r) in enumerate(classes):
                if time.time() > deadline - 5:
                    break
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_idx = np.arange(new_r, N, m)
                    gain = int(np.sum(covered[new_idx] == 0))
                    if gain == 0:
                        continue
                    classes[i][1] = new_r
                    covered[new_idx] = 1
                    new_cur = int(np.sum(covered == 0))
                    if new_cur < cur:
                        cur = new_cur
                        improved = True
                    else:
                        classes[i][1] = r
                        covered = np.zeros(N, dtype=np.uint8)
                        for mm, rr in classes:
                            covered[np.arange(rr, N, mm)] = 1
        return classes, cur

    best_classes = None
    best_uncovered = N
    c, u = greedy(divisors)
    if u < best_uncovered:
        best_classes, best_uncovered = c, u
    if time.time() < deadline - 25:
        c, u = greedy(sorted(divisors, reverse=True))
        if u < best_uncovered:
            best_classes, best_uncovered = c, u
    if best_classes is not None and time.time() < deadline - 10:
        best_classes, best_uncovered = improve(best_classes)
    return best_classes if best_classes else []
