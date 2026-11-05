import time
import bisect
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def compute_all_products(values):
        prods = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                prods.add(prod)
        return prods

    def try_add(values, x, existing_prods=None):
        pos = bisect.bisect_left(values, x)
        if pos < len(values) and values[pos] == x:
            return None

        new_tails = []
        prod = x
        new_tails.append(prod)
        for i in range(pos - 1, -1, -1):
            prod *= values[i]
            new_tails.append(prod)

        if existing_prods is None:
            existing_prods = compute_all_products(values)

        for t in new_tails:
            if t in existing_prods:
                return None

        new_values = values[:pos] + [x] + values[pos:]
        return new_values, new_tails

    # Phase 1: Greedy in increasing order
    values = []
    for x in range(2, 129):
        result = try_add(values, x)
        if result is not None:
            values = result[0]

    best_values = list(values)
    best_len = len(values)

    # Phase 2: Iterative single insertions (multiple passes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(2, 129):
            if x in values:
                continue
            result = try_add(values, x)
            if result is not None:
                values = result[0]
                improved = True
                if len(values) > best_len:
                    best_len = len(values)
                    best_values = list(values)

    # Phase 3: Local search - remove one, add two
    if time.time() < deadline:
        random.seed(42)
        current = list(best_values)
        no_improve_count = 0

        for iteration in range(30000):
            if time.time() > deadline:
                break
            if len(current) < 2:
                break

            # Remove a random element
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]
            candidate_set = set(candidate)

            available = [v for v in range(2, 129) if v not in candidate_set]
            if len(available) < 2:
                continue

            # Try to add two random elements
            found = False
            for attempt in range(10):
                pair = random.sample(available, 2)
                temp = list(candidate)
                added = 0
                for x in pair:
                    result = try_add(temp, x)
                    if result is not None:
                        temp = result[0]
                        added += 1

                if added == 2 and len(temp) > len(current):
                    current = temp
                    if len(current) > best_len:
                        best_len = len(current)
                        best_values = list(current)
                    no_improve_count = 0
                    found = True
                    break

            if not found:
                no_improve_count += 1
                if no_improve_count > 1000:
                    break

    return best_values
