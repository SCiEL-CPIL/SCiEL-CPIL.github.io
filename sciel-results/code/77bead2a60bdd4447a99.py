import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    N = 256

    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)

    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def _exp(x):
        """Safe exponential for simulated annealing."""
        if x < -100:
            return 0.0
        if x > 100:
            return 1e100
        return math.exp(x)

    def is_valid_add(s, x):
        """Check if adding x to s creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s:
                    return False
            elif x == b:
                if a in s and c in s and d in s:
                    return False
            elif x == c:
                if a in s and b in s and d in s:
                    return False
            else:
                if a in s and b in s and c in s:
                    return False
        return True

    def delta_add(s, x):
        """Change in 3-AP count if we add x to s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return delta

    def delta_remove(s, x):
        """Change in 3-AP count if we remove x from s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return -delta

    def is_valid_swap(s, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            if x_in == a:
                others = [b, c, d]
            elif x_in == b:
                others = [a, c, d]
            elif x_in == c:
                others = [a, b, d]
            else:
                others = [a, b, c]

            all_in_s = True
            x_out_in_others = False
            for y in others:
                if y not in s:
                    all_in_s = False
                    break
                if y == x_out:
                    x_out_in_others = True

            if all_in_s and not x_out_in_others:
                return False
        return True

    def greedy_construct(random_seed):
        random.seed(random_seed)
        s = set()
        available = list(range(1, N + 1))

        while available:
            candidates = []
            for x in available:
                if is_valid_add(s, x):
                    d = delta_add(s, x)
                    candidates.append((d, x))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0], reverse=True)

            top_k = min(5, len(candidates))
            if random.random() < 0.8:
                best_x = candidates[0][1]
            else:
                best_x = random.choice([c[1] for c in candidates[:top_k]])

            s.add(best_x)
            available.remove(best_x)

        return s

    def local_search_1move(s):
        """Try add and remove moves to improve 3-AP count."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                if not is_valid_add(s, x):
                    continue
                gain = delta_add(s, x)
                if gain > 0:
                    s.add(x)
                    improved = True
                    break
            if not improved:
                for x in list(s):
                    loss = delta_remove(s, x)
                    if loss >= 0:
                        s.discard(x)
                        improved = True
                        break
        return s

    def local_search_swap(s):
        """Try swap moves to improve 3-AP count."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            for x_out in list(s):
                loss = -delta_remove(s, x_out)
                if loss == 0:
                    continue
                for x_in in range(1, N + 1):
                    if x_in in s:
                        continue
                    if not is_valid_swap(s, x_out, x_in):
                        continue
                    gain = 0
                    for ap_idx in elem_to_3aps[x_in]:
                        a, b, c = three_aps[ap_idx]
                        if x_in == a:
                            if b in s and c in s and b != x_out and c != x_out:
                                gain += 1
                        elif x_in == b:
                            if a in s and c in s and a != x_out and c != x_out:
                                gain += 1
                        else:
                            if a in s and b in s and a != x_out and b != x_out:
                                gain += 1

                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_out, x_in)
            if best_swap is not None:
                x_out, x_in = best_swap
                s.remove(x_out)
                s.add(x_in)
                improved = True
        return s

    def simulated_annealing(s, t0=1.0, t1=0.01, steps=2000):
        """Simulated annealing with add/remove/swap moves."""
        s = set(s)
        current_score = count_3aps(s)
        best_s = set(s)
        best_score = current_score

        for step in range(steps):
            if time.time() > deadline:
                break
            T = t0 * (t1 / t0) ** (step / steps)

            move_type = random.random()

            if move_type < 0.3:
                candidates = [x for x in range(1, N + 1) if x not in s and is_valid_add(s, x)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_s = s | {x}
                delta = delta_add(s, x)
                new_score = current_score + delta
            elif move_type < 0.5:
                if not s:
                    continue
                x = random.choice(list(s))
                new_s = s - {x}
                delta = delta_remove(s, x)
                new_score = current_score + delta
            else:
                if not s:
                    continue
                x_out = random.choice(list(s))
                candidates_in = [x for x in range(1, N + 1) if x not in s]
                if not candidates_in:
                    continue
                x_in = random.choice(candidates_in)
                if not is_valid_swap(s, x_out, x_in):
                    continue
                loss = -delta_remove(s, x_out)
                gain = 0
                for ap_idx in elem_to_3aps[x_in]:
                    a, b, c = three_aps[ap_idx]
                    if x_in == a:
                        if b in s and c in s and b != x_out and c != x_out:
                            gain += 1
                    elif x_in == b:
                        if a in s and c in s and a != x_out and c != x_out:
                            gain += 1
                    else:
                        if a in s and b in s and a != x_out and b != x_out:
                            gain += 1
                delta = gain - loss
                new_s = (s - {x_out}) | {x_in}
                new_score = current_score + delta

            if new_score > current_score or random.random() < _exp((new_score - current_score) / T):
                s = new_s
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_s = set(s)

        return best_s

    best_s = set()
    best_score = -1

    # Phase 1: Multiple greedy constructions with local search
    num_restarts = 25
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break

        s = greedy_construct(i)
        score = count_3aps(s)

        if score > best_score:
            best_score = score
            best_s = set(s)

        s = local_search_1move(s)
        score = count_3aps(s)

        if score > best_score:
            best_score = score
            best_s = set(s)

        s = local_search_swap(s)
        score = count_3aps(s)

        if score > best_score:
            best_score = score
            best_s = set(s)

        # Short SA on this solution
        if time.time() < deadline - 10:
            s = simulated_annealing(s, t0=0.5, t1=0.01, steps=500)
            score = count_3aps(s)

            if score > best_score:
                best_score = score
                best_s = set(s)

    # Phase 2: Final high-precision polish on the best solution
    if best_s and time.time() < deadline - 3:
        remaining = deadline - time.time() - 3
        # Use more iterations with lower temperature for fine-tuning
        steps = int(remaining * 150)  # Scale steps to available time
        s = simulated_annealing(best_s, t0=0.2, t1=0.001, steps=steps)
        score = count_3aps(s)

        if score > best_score:
            best_score = score
            best_s = set(s)

        # One more very low temperature pass
        if time.time() < deadline - 1:
            s = simulated_annealing(best_s, t0=0.05, t1=0.0001, steps=500)
            score = count_3aps(s)
            if score > best_score:
                best_score = score
                best_s = set(s)

    return sorted(best_s)
