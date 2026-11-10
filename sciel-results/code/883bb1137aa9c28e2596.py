import time
import random
import bisect

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin before 90s kill

    def is_valid(seq):
        """
        Check if the sequence has distinct products for all non-empty consecutive blocks.
        seq must be sorted (strictly increasing).
        """
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    # 1. Greedy Initialization
    S = []
    for x in range(2, 129):
        new_S = S + [x]
        if is_valid(new_S):
            S = new_S

    best_S = S[:]
    best_len = len(S)

    def try_insert(S, x):
        """Try to insert x into sorted S; return new list if valid, else None."""
        idx = bisect.bisect_left(S, x)
        new_S = S[:idx] + [x] + S[idx:]
        if is_valid(new_S):
            return new_S
        return None

    def try_remove_add(S, remove_val, add_vals):
        """Remove remove_val from S, add all add_vals, sort, check validity."""
        new_S = [v for v in S if v != remove_val]
        new_S.extend(add_vals)
        new_S.sort()
        if is_valid(new_S):
            return new_S
        return None

    def try_swap(S, remove_val, add_val):
        """Remove remove_val, add add_val, sort, check validity."""
        new_S = [v for v in S if v != remove_val]
        new_S.append(add_val)
        new_S.sort()
        if is_valid(new_S):
            return new_S
        return None

    # Main search loop
    while time.time() - start_time < limit:
        improved = False

        # Step A: Try to insert any single missing number (greedy + random order)
        missing = [x for x in range(2, 129) if x not in S]
        random.shuffle(missing)

        for x in missing:
            if time.time() - start_time > limit:
                break
            new_S = try_insert(S, x)
            if new_S is not None:
                S = new_S
                improved = True
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                break

        if time.time() - start_time > limit:
            break

        if not improved:
            # Step B: Try 1-out, 2-in moves to increase length by 1
            if len(S) < 127:
                moves_tried = 0
                max_moves = 100
                while moves_tried < max_moves and time.time() - start_time < limit:
                    moves_tried += 1

                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if len(candidates) < 2:
                        continue

                    found = False
                    for _ in range(20):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1

                        new_S = try_remove_add(S, y, [x1, x2])
                        if new_S is not None:
                            S = new_S
                            improved = True
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            found = True
                            break

                    if found:
                        break

                if improved:
                    continue

            # Step C: Try 2-out, 2-in moves (same length, different composition)
            if len(S) >= 4:
                swap_tried = 0
                while swap_tried < 40 and time.time() - start_time < limit:
                    swap_tried += 1

                    y1, y2 = random.sample(S, 2)
                    candidates = [x for x in range(2, 129) if x not in S and x != y1 and x != y2]
                    if len(candidates) < 2:
                        continue

                    for _ in range(15):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1

                        new_S = try_remove_add(S, y1, [x2])
                        if new_S is not None:
                            # Remove y2 from new_S and add x1
                            new_S2 = [v for v in new_S if v != y2]
                            new_S2.append(x1)
                            new_S2.sort()
                            if is_valid(new_S2):
                                S = new_S2
                                break

                if time.time() - start_time > limit:
                    break

            # Step D: Try 1-out, 1-in swap moves to explore alternative compositions
            if len(S) >= 2:
                swap_tried = 0
                while swap_tried < 30 and time.time() - start_time < limit:
                    swap_tried += 1

                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if not candidates:
                        continue

                    x = random.choice(candidates)
                    new_S = try_swap(S, y, x)
                    if new_S is not None:
                        S = new_S
                        break

        # If we've been stuck for a while, do a more aggressive random restart of the local search
        # by trying many random 1-out/1-in moves in sequence
        if time.time() - start_time < limit:
            stuck_tries = 0
            while stuck_tries < 50 and time.time() - start_time < limit:
                stuck_tries += 1
                y = random.choice(S)
                candidates = [x for x in range(2, 129) if x not in S and x != y]
                if not candidates:
                    break
                x = random.choice(candidates)
                new_S = try_swap(S, y, x)
                if new_S is not None:
                    S = new_S
                    # After a successful swap, immediately try to insert
                    missing2 = [x for x in range(2, 129) if x not in S]
                    random.shuffle(missing2)
                    for x2 in missing2:
                        if time.time() - start_time > limit:
                            break
                        new_S2 = try_insert(S, x2)
                        if new_S2 is not None:
                            S = new_S2
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            break

    return best_S
