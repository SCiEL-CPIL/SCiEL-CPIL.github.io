import time
import random

def _pf(n):
    if n <= 1:
        return set()
    s = set()
    while n % 2 == 0:
        s.add(2); n //= 2
    while n % 3 == 0:
        s.add(3); n //= 3
    f = 5
    while f * f <= n:
        while n % f == 0:
            s.add(f); n //= f
        f += 2
    if n > 1:
        s.add(n)
    return s

def construct():
    deadline = time.time() + 80
    N = 12
    random.seed(7)

    def score(xs):
        pr = {2, 3}
        for i in range(N):
            xi = xs[i]
            for j in range(i + 1, N):
                t = xi + xs[j]
                while t % 2 == 0:
                    t //= 2
                while t % 3 == 0:
                    t //= 3
                if t > 1:
                    pr |= _pf(t)
        return len(pr)

    best_x = list(range(1, N + 1))
    best_s = score(best_x)

    for base in range(1, 200, 3):
        cand = [base + i for i in range(N)]
        if max(cand) <= 1666:
            s = score(cand)
            if s < best_s:
                best_s = s
                best_x = cand[:]

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        xs = best_x[:]
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old = xs[i]
            for _ in range(80):
                if random.random() < 0.3:
                    new_x = random.randint(1, 1666)
                else:
                    new_x = old + random.randint(-50, 50)
                if new_x < 1 or new_x > 1666 or new_x in xs:
                    continue
                xs[i] = new_x
                s = score(xs)
                if s < best_s:
                    best_s = s
                    best_x = xs[:]
                    improved = True
                xs[i] = old

    return [6 * x for x in best_x]
# EVOLVE-BLOCK-END
