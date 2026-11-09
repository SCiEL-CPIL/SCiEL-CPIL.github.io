import time
import random

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    Precompute all conflict triples (a, b, c) where a, b, c are distinct,
    1 <= a,b,c <= 180, and 1/a = 1/b + 1/c.
    For each pair {b,c} (b<c, both in [1,180]), a = bc/(b+c). If a is integer,
    1<=a<=180, and a distinct from b,c, then (a,b,c) is a conflict.
    Returns (conflicts, pair_to_a, elem_to_pairs).
    """
    conflicts = []
    pair_to_a = {}
    elem_to_pairs = {i: [] for i in range(1, 181)}
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    conflicts.append((a, b, c))
                    pair_to_a[(b, c)] = a
                    elem_to_pairs[a].append((b, c))
                    elem_to_pairs[b].append((a, c))
                    elem_to_pairs[c].append((a, b))
    return conflicts, pair_to_a, elem_to_pairs

def _can_add(current, x, pair_to_a):
    """Check if adding x to current (a set) creates any conflict."""
    for (y, z) in elem_to_pairs[x]:
        if y in current and z in current:
            return False
    return True

def _greedy_build(order, pair_to_a, deadline):
    """Greedily build a valid set by adding elements in the given order."""
    current = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in current:
            continue
        if _can_add(current, x, pair_to_a):
            current.add(x)
    return current

def _conflict_degree(x, pair_to_a):
    """Number of pairs involving x (proxy for how 'constraining' x is)."""
    return len(elem_to_pairs[x])

def _local_search(current_set, pair_to_a, elem_to_pairs, all_elements, deadline,
                  max_rounds=400, sa=False, T0=1.0, T_end=0.001):
    """
    Local search with:
      - Add any missing element
      - Remove 1, greedy add
      - Remove 2, greedy add
      - 2-swap (remove 2, add 2)
      - Optional simulated annealing acceptance of non-improving moves
    """
    current = set(current_set)
    best = set(current)
    best_size = len(current)
    all_set = set(all_elements)

    T = T0
    n_steps = max(1, max_rounds)
    step = 0
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        step += 1
        if sa:
            T = T0 * (T_end / T0) ** (step / n_steps)
        improved = False

        # 1) Try to add any missing element
        missing = [x for x in all_elements if x not in current]
        random.shuffle(missing)
        added_any = False
        for m in missing:
            if time.time() > deadline:
                break
            if _can_add(current, m, pair_to_a):
                current.add(m)
                added_any = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
        if added_any:
            improved = True
            continue

        # 2) Remove 1, greedy add
        in_set = list(current)
        random.shuffle(in_set)
        for x in in_set:
            if time.time() > deadline:
                break
            temp = current - {x}
            missing_after = [y for y in all_elements if y not in temp]
            random.shuffle(missing_after)
            for y in missing_after:
                if time.time() > deadline:
                    break
                if _can_add(temp, y, pair_to_a):
                    temp.add(y)
            if len(temp) > len(current):
                current = temp
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                break
        if improved:
            continue

        # 3) Remove 2, greedy add
        in_set = list(current)
        random.shuffle(in_set)
        n = len(in_set)
        for i in range(n):
            if time.time() > deadline:
                break
            x1 = in_set[i]
            for j in range(i + 1, n):
                x2 = in_set[j]
                temp = current - {x1, x2}
                missing_after = [y for y in all_elements if y not in temp]
                random.shuffle(missing_after)
                for y in missing_after:
                    if time.time() > deadline:
                        break
                    if _can_add(temp, y, pair_to_a):
                        temp.add(y)
                if len(temp) > len(current):
                    current = temp
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    break
            if improved:
                break
        if improved:
            continue

        # 4) 2-swap: remove 2, add 2 (net 0), then try to add more
        in_set = list(current)
        random.shuffle(in_set)
        missing = [x for x in all_elements if x not in current]
        random.shuffle(missing)
        n = len(in_set)
        done = False
        for i in range(n):
            if time.time() > deadline:
                break
            x1 = in_set[i]
            for j in range(i + 1, n):
                x2 = in_set[j]
                temp = current - {x1, x2}
                added = 0
                for y in missing:
                    if added >= 2:
                        break
                    if _can_add(temp, y, pair_to_a):
                        temp.add(y)
                        added += 1
                if added < 2:
                    continue
                # now try to add more
                missing_after = [y for y in all_elements if y not in temp]
                random.shuffle(missing_after)
                for y in missing_after:
                    if time.time() > deadline:
                        break
                    if _can_add(temp, y, pair_to_a):
                        temp.add(y)
                if len(temp) > len(current):
                    current = temp
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    done = True
                    break
                elif sa and len(temp) == len(current) and random.random() < T:
                    # accept a neutral move to escape local optimum
                    current = temp
                    done = True
                    break
            if done:
                break
        if improved or (sa and done):
            continue

        # No improvement this round; with SA we may continue, otherwise break
        if not sa:
            break

    return best

def _verify(s_set, pair_to_a):
    """Final verification: no three distinct elements form a conflict."""
    for x in s_set:
        for (y, z) in elem_to_pairs[x]:
            if y in s_set and z in s_set and x != y and x != z:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    conflicts, pair_to_a, elem_to_pairs = _precompute_conflicts()
    all_elements = list(range(1, 181))

    best_set = set()
    best_size = 0

    def consider(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    # Greedy: descending
    consider(_greedy_build(all_elements[::-1], pair_to_a, deadline))
    # Greedy: ascending
    consider(_greedy_build(all_elements, pair_to_a, deadline))
    # Greedy: by conflict degree (most constrained first)
    by_deg = sorted(all_elements, key=lambda x: _conflict_degree(x, pair_to_a))
    consider(_greedy_build(by_deg, pair_to_a, deadline))
    consider(_greedy_build(by_deg[::-1], pair_to_a, deadline))

    # Random greedy restarts
    for restart in range(40):
        if time.time() > deadline:
            break
        random.seed(1000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        consider(_greedy_build(shuffled, pair_to_a, deadline))

    # Local search from the best set
    consider(_local_search(best_set, pair_to_a, elem_to_pairs, all_elements, deadline, max_rounds=300))

    # Local search from several greedy solutions (deterministic + random)
    starts = [all_elements[::-1], all_elements, by_deg, by_deg[::-1]]
    for restart in range(20):
        random.seed(2000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        starts.append(shuffled)

    for idx, order in enumerate(starts):
        if time.time() > deadline:
            break
        g = _greedy_build(order, pair_to_a, deadline)
        ls = _local_search(g, pair_to_a, elem_to_pairs, all_elements, deadline, max_rounds=200)
        consider(ls)

    # Simulated annealing from a dense random start and from best
    if time.time() < deadline:
        random.seed(5000)
        dense = set(all_elements)
        consider(_local_search(dense, pair_to_a, elem_to_pairs, all_elements, deadline,
                               max_rounds=400, sa=True, T0=2.0, T_end=0.001))
    if time.time() < deadline:
        consider(_local_search(best_set, pair_to_a, elem_to_pairs, all_elements, deadline,
                               max_rounds=400, sa=True, T0=1.5, T_end=0.001))

    # Final verification; if somehow invalid, fall back to a safe greedy
    if not _verify(best_set, pair_to_a):
        fallback = _greedy_build(all_elements[::-1], pair_to_a, deadline)
        if len(fallback) > 0:
            best_set = fallback

    return list(best_set)

# EVOLVE-BLOCK-END
