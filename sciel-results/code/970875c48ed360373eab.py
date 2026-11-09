import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n_max = 256

    # Precompute gcd table for speed
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(2, n_max + 1):
        for j in range(i, n_max + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid(S):
        """Full O(n^3) validation."""
        S_list = sorted(S)
        m = len(S_list)
        if m < 3:
            return True
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                d = a // gcd_table[a][b]
                if d == 1:
                    # a | b*c for any c, and there exist c distinct from a,b
                    return False
                # Check if any c in S_list (c != a, c != b) is divisible by d
                for k in range(m):
                    if k == i or k == j:
                        continue
                    if S_list[k] % d == 0:
                        return False
        return True

    def can_add(x, S_list, S_set):
        """Check if S ∪ {x} is valid. Only need to check triples involving x."""
        m = len(S_list)
        if m < 2:
            return True
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                # Check a | x*b
                d1 = a // gcd_table[a][x]
                if d1 == 1:
                    return False
                if b % d1 == 0:
                    return False
                # Check b | x*a
                d2 = b // gcd_table[b][x]
                if d2 == 1:
                    return False
                if a % d2 == 0:
                    return False
                # Check x | a*b
                d3 = x // gcd_table[x][a]
                if d3 == 1:
                    return False
                if b % d3 == 0:
                    return False
                d4 = x // gcd_table[x][b]
                if d4 == 1:
                    return False
                if a % d4 == 0:
                    return False
        return True

    # Phase 1: Start with all large numbers [129, 256]
    S = list(range(129, 257))
    S_set = set(S)
    S_list = S[:]

    # Phase 2: Greedily add small numbers 2..128
    # Try in increasing order; also try in a few different orders
    best_S = None
    best_size = len(S_list)

    if time.time() < deadline:
        if is_valid(S_list):
            best_S = S_list[:]

    # Try adding small numbers in increasing order
    for x in range(2, 129):
        if time.time() > deadline - 5:
            break
        if x in S_set:
            continue
        if can_add(x, S_list, S_set):
            S_list.append(x)
            S_set.add(x)
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S = S_list[:]

    # Phase 3: Try a second pass with different ordering (decreasing small numbers)
    if time.time() < deadline - 10:
        S2 = list(range(129, 257))
        S2_set = set(S2)
        S2_list = S2[:]
        for x in range(128, 1, -1):
            if time.time() > deadline - 5:
                break
            if x in S2_set:
                continue
            if can_add(x, S2_list, S2_set):
                S2_list.append(x)
                S2_set.add(x)
                if len(S2_list) > best_size:
                    best_size = len(S2_list)
                    best_S = S2_list[:]

    # Phase 4: Try starting with [120, 256] and adding 2..119
    if time.time() < deadline - 10:
        S3 = list(range(120, 257))
        S3_set = set(S3)
        S3_list = S3[:]
        if is_valid(S3_list):
            if len(S3_list) > best_size:
                best_size = len(S3_list)
                best_S = S3_list[:]
        for x in range(2, 120):
            if time.time() > deadline - 5:
                break
            if x in S3_set:
                continue
            if can_add(x, S3_list, S3_set):
                S3_list.append(x)
                S3_set.add(x)
                if len(S3_list) > best_size:
                    best_size = len(S3_list)
                    best_S = S3_list[:]

    # Phase 5: Try [100, 256] + greedy add 2..99
    if time.time() < deadline - 10:
        S4 = list(range(100, 257))
        S4_set = set(S4)
        S4_list = S4[:]
        if is_valid(S4_list):
            if len(S4_list) > best_size:
                best_size = len(S4_list)
                best_S = S4_list[:]
        for x in range(2, 100):
            if time.time() > deadline - 5:
                break
            if x in S4_set:
                continue
            if can_add(x, S4_list, S4_set):
                S4_list.append(x)
                S4_set.add(x)
                if len(S4_list) > best_size:
                    best_size = len(S4_list)
                    best_S = S4_list[:]

    # Phase 6: Local improvement - try to swap out a small element for another
    if best_S is not None and time.time() < deadline - 10:
        current = best_S[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(2, 257):
                if x in current_set:
                    continue
                # Try adding x by removing some element y
                for y in current[:]:
                    if y >= 129:
                        continue  # don't remove large elements
                    new_set = current_set - {y} | {x}
                    new_list = [z for z in current if z != y] + [x]
                    if is_valid(new_list):
                        current = new_list
                        current_set = new_set
                        if len(current) > best_size:
                            best_size = len(current)
                            best_S = current[:]
                        improved = True
                        break
                if time.time() > deadline - 5:
                    break

    # Final validation
    if best_S is None:
        # Fallback: all primes
        is_prime = [True] * (n_max + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n_max**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n_max + 1, i):
                    is_prime[j] = False
        best_S = [i for i in range(2, n_max + 1) if is_prime[i]]

    # Final check
    if not is_valid(best_S):
        # Fallback to primes
        is_prime = [True] * (n_max + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n_max**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n_max + 1, i):
                    is_prime[j] = False
        best_S = [i for i in range(2, n_max + 1) if is_prime[i]]

    return list(best_S)
