import time
import random

def construct():
    n = 256
    t_start = time.time()
    DEADLINE = t_start + 85.0  # safety margin before 90s limit

    def can_add(x, S_list):
        """Check if x can be added to S_list (sorted) without violating constraints.
        Only need to check triples where x is the smallest element (a=x),
        since x < all elements in S_list when we add smaller elements.
        For general use, we check all three cases."""
        # Case 1: x is 'a' (smallest). Check x !| b+c for all b,c in S with b,c > x.
        # For our greedy (adding smaller to larger), x < min(S), so all b,c > x.
        # Count residues of S elements mod x, check for complementary pairs.
        if x <= 0:
            return False
        residues = {}
        for v in S_list:
            if v > x:  # b and c must be > a=x
                r = v % x
                residues[r] = residues.get(r, 0) + 1
        for r, cnt in residues.items():
            if r == 0:
                if cnt >= 2:
                    return False
            else:
                comp = (x - r) % x
                if comp in residues:
                    if comp == r:
                        if cnt >= 2:
                            return False
                    else:
                        if residues[comp] >= 1:
                            return False
        return True

    def can_add_general(x, S_list):
        """General can_add for local search where x might be any size."""
        # Build new sorted list and check all triples involving x
        S_new = sorted(S_list + [x])
        L = len(S_new)
        # Find index of x
        idx_x = S_new.index(x)
        for i in range(L):
            a = S_new[i]
            if a == x and i < idx_x:
                continue  # duplicate x shouldn't happen
            for j in range(i + 1, L):
                b = S_new[j]
                for k in range(j, L):
                    c = S_new[k]
                    if (b + c) % a == 0:
                        # Check if this triple involves x
                        if a == x or b == x or c == x:
                            return False
        return True

    def is_valid(S_list):
        """Full validation of sorted list."""
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j, L):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    best_solution = list(range(171, 257))
    best_size = len(best_solution)

    # Strategy 1: Try multiple base intervals with greedy addition
    for k in range(155, 180):
        if time.time() > DEADLINE:
            break
        S = list(range(k, n + 1))
        S_sorted = sorted(S)
        # Add elements from k-1 down to 1
        for x in range(k - 1, 0, -1):
            if time.time() > DEADLINE:
                break
            if can_add(x, S_sorted):
                S.append(x)
                S_sorted = sorted(S)
        if len(S) > best_size:
            best_size = len(S)
            best_solution = list(S)

    # Strategy 2: Local search on best solution
    random.seed(42)
    S_cur = list(best_solution)
    S_cur_sorted = sorted(S_cur)

    # Phase 1: Try to add any missing element (greedy fill)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        S_cur_set = set(S_cur)
        for x in range(1, n + 1):
            if time.time() > DEADLINE:
                break
            if x not in S_cur_set:
                if can_add_general(x, S_cur_sorted):
                    S_cur.append(x)
                    S_cur_set.add(x)
                    S_cur_sorted = sorted(S_cur)
                    changed = True
                    break

    if len(S_cur) > best_size:
        best_size = len(S_cur)
        best_solution = list(S_cur)

    # Phase 2: Local search with swaps
    for restart in range(30):
        if time.time() > DEADLINE:
            break

        S_cur = list(best_solution)
        if restart > 0:
            # Perturb: remove 1-2 random elements
            for _ in range(random.randint(1, 2)):
                if len(S_cur) > 10:
                    idx = random.randint(0, len(S_cur) - 1)
                    S_cur.pop(idx)

        S_cur_set = set(S_cur)
        S_cur_sorted = sorted(S_cur)

        # Try to add missing elements
        for x in range(1, n + 1):
            if time.time() > DEADLINE:
                break
            if x not in S_cur_set:
                if can_add_general(x, S_cur_sorted):
                    S_cur.append(x)
                    S_cur_set.add(x)
                    S_cur_sorted = sorted(S_cur)

        # Try swaps: remove one, add another
        for _ in range(20):
            if time.time() > DEADLINE:
                break
            if len(S_cur) <= 10:
                break
            S_cur_set = set(S_cur)
            S_cur_sorted = sorted(S_cur)

            # Pick random element to remove
            y = random.choice(S_cur)
            S_temp = [v for v in S_cur if v != y]
            S_temp_set = set(S_temp)
            S_temp_sorted = sorted(S_temp)

            # Try to add an element
            best_add = None
            for x in range(1, n + 1):
                if time.time() > DEADLINE:
                    break
                if x not in S_temp_set and x != y:
                    if can_add_general(x, S_temp_sorted):
                        best_add = x
                        break

            if best_add is not None:
                S_cur = S_temp + [best_add]

        if len(S_cur) > best_size:
            best_size = len(S_cur)
            best_solution = list(S_cur)

    S_final = sorted(best_solution)
    if not is_valid(S_final):
        S_final = list(range(171, 257))

    return S_final
