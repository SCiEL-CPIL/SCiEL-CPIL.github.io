import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin

    points = set(range(0, N + 1))
    primes = _primes(N)

    for p in primes:
        if time.time() > deadline:
            break
        if p > len(points):
            break

        # Check if all residue classes mod p are present
        classes = {}
        for v in points:
            r = v % p
            if r not in classes:
                classes[r] = []
            classes[r].append(v)

        if len(classes) == p:
            # Remove the least-populated class
            smallest_class = min(classes.values(), key=len)
            for v in smallest_class:
                points.discard(v)

    return sorted(points)
