import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Handle the specific task case: q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        neighbors = [[] for _ in range(total_words)]
        
        # Precompute neighbors (ball of radius 1) for all words
        # We use base-3 representation where index = sum(w[i] * 3**i)
        for i in range(total_words):
            w = [0] * n
            temp = i
            for k in range(n):
                w[k] = temp % 3
                temp //= 3
            
            neighbors[i].append(i)
            for pos in range(n):
                orig = w[pos]
                for val in range(3):
                    if val == orig:
                        continue
                    w[pos] = val
                    new_idx = 0
                    mult = 1
                    for k in range(n):
                        new_idx += w[k] * mult
                        mult *= 3
                    neighbors[i].append(new_idx)
                w[pos] = orig

        # Greedy Set Cover with efficient counting
        covered = [False] * total_words
        # uncovered_count[i] = number of neighbors of i that are currently uncovered
        uncovered_count = [len(neighbors[i]) for i in range(total_words)]
        code = []
        
        while True:
            if time.time() > deadline:
                break
            
            # Find word with max uncovered_count
            max_count = -1
            max_idx = -1
            for i in range(total_words):
                if uncovered_count[i] > max_count:
                    max_count = uncovered_count[i]
                    max_idx = i
                    if max_count == 0:
                        break
            
            if max_count == 0:
                break
            
            code.append(max_idx)
            
            # Update covered status and counts
            for j in neighbors[max_idx]:
                if not covered[j]:
                    covered[j] = True
                    for k in neighbors[j]:
                        uncovered_count[k] -= 1

        # Convert indices back to words
        result = []
        for i in code:
            w = [0] * n
            temp = i
            for k in range(n):
                w[k] = temp % 3
                temp //= 3
            result.append(w)
        return result
    else:
        # Fallback: Direct sum of tetracode if possible, else trivial
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial upper bound: all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]
