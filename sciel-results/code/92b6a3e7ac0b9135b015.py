import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V_set = set(elements)
        n = len(V_set)

        # Build hypergraph adjacency: for each v, list of (u, w) such that {v,u,w} is a sum-free violation
        # i.e., v + u = w or v + w = u or u + w = v
        adj = {v: [] for v in V_set}
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            for j in range(i + 1, n):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))

        # Degree: number of hyperedges containing v
        deg = {v: len(adj[v]) for v in V_set}

        def check_conflict(v, S):
            """Return True if v can be added to S without creating a sum-free violation."""
            for (u, w) in adj[v]:
                if u == v:
                    if w in S:
                        return False
                else:
                    if u in S and w in S:
                        return False
            return True

        def greedy_fill(order, S=None):
            """Greedy fill following given order."""
            if S is None:
                S = set()
            for v in order:
                if v not in S and check_conflict(v, S):
                    S.add(v)
            return S

        rng = random.Random(seed_off)

        # --- Multiple greedy initializations ---
        candidates = []

        # 1. Odds (always sum-free)
        S_odds = set(x for x in V_set if x % 2 != 0)
        candidates.append(S_odds)

        # 2. Greedy by degree (ascending - low degree first)
        order_deg = sorted(V_set, key=lambda x: (deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg))

        # 3. Greedy by degree (descending)
        order_deg_desc = sorted(V_set, key=lambda x: (-deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg_desc))

        # 4. Greedy by absolute value (ascending)
        order_abs = sorted(V_set, key=lambda x: abs(x))
        candidates.append(greedy_fill(order_abs))

        # 5. Greedy by absolute value (descending)
        order_abs_desc = sorted(V_set, key=lambda x: -abs(x))
        candidates.append(greedy_fill(order_abs_desc))

        # 6. Suffix: largest elements where 2*min > max
        S_sorted = sorted(V_set, reverse=True)
        suffix = set()
        for k in range(len(S_sorted)):
            if 2 * S_sorted[k] > S_sorted[0]:
                suffix = set(S_sorted[k:])
                break
        if suffix:
            candidates.append(suffix)

        # 7-12. Random orderings
        for _ in range(6):
            order = list(V_set)
            rng.shuffle(order)
            candidates.append(greedy_fill(order))

        # Pick best candidate
        best_S = max(candidates, key=len)

        # --- Local search: 1-for-k swaps ---
        # Try removing one element and greedily re-adding others
        while time.time() - t_start < budget:
            improved = False
            best_copy = set(best_S)
            elements_list = list(best_copy)

            # Try removing each element
            for x in elements_list:
                if time.time() - t_start >= budget:
                    break
                trial = set(best_copy)
                trial.discard(x)

                # Try to add elements from V_set \ trial in order of low degree
                for y in sorted(V_set - trial, key=lambda v: deg[v]):
                    if check_conflict(y, trial):
                        trial.add(y)

                if len(trial) > len(best_S):
                    best_S = trial
                    improved = True
                    break

            if not improved:
                # Try 2-for-k swaps: remove two elements
                found = False
                for i in range(len(elements_list)):
                    if time.time() - t_start >= budget:
                        break
                    for j in range(i + 1, len(elements_list)):
                        trial = set(best_S)
                        trial.discard(elements_list[i])
                        trial.discard(elements_list[j])

                        for y in sorted(V_set - trial, key=lambda v: deg[v]):
                            if check_conflict(y, trial):
                                trial.add(y)

                        if len(trial) > len(best_S):
                            best_S = trial
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Try perturbation: remove 1-3 random elements, re-greedy
                    perturbed_any = False
                    for _ in range(15):
                        if time.time() - t_start >= budget:
                            break
                        perturbed = set(best_S)
                        num_remove = random.randint(1, 3)
                        for _ in range(num_remove):
                            if perturbed:
                                perturbed.discard(random.choice(list(perturbed)))
                        # Re-greedy on remaining
                        remaining = [x for x in V_set if x not in perturbed]
                        random.shuffle(remaining)
                        for y in remaining:
                            if check_conflict(y, perturbed):
                                perturbed.add(y)
                        if len(perturbed) > len(best_S):
                            best_S = perturbed
                            improved = True
                            perturbed_any = True
                            break
                    if not perturbed_any and not improved:
                        break

        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803)
    ]

    results = []
    start = time.time()

    for i, S in enumerate(instances):
        elapsed = time.time() - start
        # Allocate more time to larger instances
        if i == 0:
            budget = min(30, DEADLINE - elapsed - 5)
        elif i == 1:
            budget = min(45, DEADLINE - elapsed - 5)
        else:
            budget = min(DEADLINE - elapsed - 5, 100)

        if budget <= 0:
            # Fallback: odds
            results.append([x for x in S if x % 2 != 0])
            continue

        res = solve_instance(S, seed_off=1000 + i, budget=budget)
        results.append(res)

    return results
